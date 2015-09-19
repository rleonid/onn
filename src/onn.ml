
open Lacaml.D
open Bigarray


(***** Initialization ******)
let vi = ref (fun n -> Vec.make0 n)

let seed_ref = ref (Array1.of_array Int32 Fortran_layout [| 0l;0l;0l;0l |])

let initialize_random_vec_gen seed =
  let seed1 = Int32.of_int seed in
  let seed2 = Int32.add seed1 1l in
  let seed3 = Int32.add seed2 1l in
  let seed4 = Int32.add (Int32.mul (Int32.add seed3 1l) 2l) 1l in
  seed_ref := Array1.of_array Int32 Fortran_layout [|seed1; seed2; seed3; seed4 |];
  let f n = larnv ~idist:`Normal ~iseed:(!seed_ref) ~n () in
  vi := f

let () = initialize_random_vec_gen 0

(***** Generating random permutations *************)
let shuffle_gen n =
  let arr = Array.init n (fun i -> i) in
  for i = 0 to n - 2 do
    arr.(i) <- arr.(i) + Random.int (n - i)
  done;
  arr

let permutation_gen n =
  shuffle_gen n
  |> Array.map (fun i -> Int32.of_int (i + 1))
  |> Array1.of_array Int32 Fortran_layout



(****** Compilation
      description -> evaluable/trainable data structures.
*)
type hidden =
  { nonlinearity : Nonlinearity.t
  ; weights      : mat
  ; bias         : vec
  ; weight_input : vec
  ; p_activation : vec
  ; bias_e       : vec
  ; weights_e    : mat
  }

let hidden nonlinearity weights bias =
  let m = Mat.dim1 weights in
  let n = Mat.dim2 weights in
  { nonlinearity
  ; weights
  ; bias
  ; weight_input = Vec.make0 m
  ; p_activation = Vec.make0 n
  ; bias_e       = Vec.make0 m
  ; weights_e    = Mat.make0 m n
  }

let rec repr (acc, l) = function
  | Desc.Input n             -> n, acc, l
  | Desc.Hidden (nl, n, pl)  ->
      let b = !vi n in
      let m = Desc.to_size pl in
      let w = Mat.of_col_vecs (Array.init m (fun _ -> !vi n)) in
      let h = hidden nl w b  in
      repr (h :: acc, l + 1) pl

type t =
  { input_size    : int
  ; hidden_layers : hidden array
  ; output_size   : int
  }

let compile desc =
  let input_size, as_lst, len = repr ([],0) desc in
  let hidden_layers = Array.of_list as_lst in
  let output_size   = Vec.dim hidden_layers.(len - 1).bias in
  { output_size; hidden_layers; input_size}

(* pla -> previous layers activation *)
let apply pla hl =
  ignore (copy ~y:hl.p_activation pla);
  gemv ~y:(copy ~y:hl.weight_input hl.bias) hl.weights pla
  |> Nonlinearity.apply hl.nonlinearity

let eval t v_0 =
  if Vec.dim v_0 <> t.input_size then
    raise (Invalid_argument "improper size.")
  else
    Array.fold_left apply v_0 t.hidden_layers

let backprop_l iteration hl prev_error =
  let i_f = float iteration in
  let ifi = 1. /. i_f in
  let sz = (Nonlinearity.deriv hl.nonlinearity) hl.weight_input in
  let dl = Vec.mul prev_error sz in
  let () = (* update bias error *)
    scal (1. -. ifi) hl.bias_e;
    axpy ~alpha:ifi dl hl.bias_e
  in
  let () = (* update weight error *)
    Mat.scal (1. -. ifi) hl.weights_e;
    ignore (ger ~alpha:ifi dl hl.p_activation hl.weights_e)
  in
  gemv ~trans:`T hl.weights dl

(* iteration which training datum (in mini batch) are we updating
  to allow implace, (online) averaging of errors. *)
let backprop iteration cost_error t =
  let bp = backprop_l iteration in
  Array.fold_right bp t.hidden_layers cost_error

let rmse_cost y y_hat = (Vec.ssqr_diff y y_hat) /. 2.
(* This can be a source of confusion *)
let rmse_cdf ~y ~y_hat = Vec.sub y_hat y

(* The 'errors' are already averaged! *)
let assign_errors learning_rate t =
  let alpha = -1.0 *. learning_rate in
  Array.iter (fun hl ->
    axpy ~alpha hl.bias_e hl.bias;
    Mat.axpy ~alpha hl.weights_e hl.weights)
    t.hidden_layers

let train_on training_offset td learning_rate cdf t =
  let ws = Vec.make0 training_offset in
  let e = eval t in
  for i = 1 to Mat.dim2 td do
    let example = Mat.col td i in
    let training = copy ~y:ws ~n:training_offset example in
    let y_hat   = e training in
    let y       = copy ~ofsx:(training_offset + 1) example in
    (*let _c      = cost y y_hat in *)
    let costd   = cdf ~y ~y_hat in
    let _finald = backprop i costd t in
    assign_errors learning_rate t
  done

let sgd_epoch training_offset td batch_size learning_rate c t =
  let td_size = Mat.dim2 td in
  let perm = permutation_gen td_size in
  let () = lapmt td perm in
  let num_bat = td_size / batch_size - 1 in
  for i = 0 to num_bat do
    (* Is this copy faster than Mat.of_cols (Mat.copy ...)? *)
    let epoch_td = lacpy ~ac:(i * batch_size + 1) ~n:batch_size td in
    train_on training_offset epoch_td learning_rate c t
  done

let sgd training_offset training_data ~epochs ~batch_size ~learning_rate
  ?(report=(fun _ -> ())) c t =
  for i = 1 to epochs do
    sgd_epoch training_offset training_data batch_size learning_rate c t;
    report t
  done

let split_validation ?seed size td =
  let () = match seed with | None -> () | Some s -> Random.init s in
  let n  = Mat.dim2 td in
  let permutation = permutation_gen n in
  lapmt td permutation;
  let t = lacpy ~n:(n - size) td in
  let v = lacpy ~ac:(n - size + 1) td in
  t, v

let max_idx_in m =
  let _, mx, _ =
    Vec.fold (fun (i,j,m) v -> if v > m then (i+1,i,v) else (i+1,j,m))
      (1, 0, neg_infinity) m
  in
  mx

let correct t y_p y =
  let i = max_idx_in y_p in
  Array1.get y i = 1.0

let report_accuracy training_offset d =
  let ws = Vec.make0 training_offset in
  let m = Mat.dim2 d in
  (fun t ->
    let e = eval t in
    let nc =
      Mat.fold_cols (fun a c ->
        let training = copy ~y:ws ~n:training_offset c in
        let y_hat    = e training in
        let y        = copy ~ofsx:(training_offset + 1) c in
        if correct t y_hat y then a + 1 else a) 0 d
    in
    (nc, m))

let td_vd_ref = ref None

let do_it ?cache ~batch_size ~hidden_layers ~epochs ~learning_rate =
  let td, vd =
    match !td_vd_ref with
    | None ->
      let d = Load_mnist.data ?cache `Train in
      let s = split_validation 10000 d in
      td_vd_ref := Some s;
      s
    | Some p -> p
  in
  let t = compile (Mnist.desc hidden_layers) in
  sgd Mnist.input_size td
    ~epochs
    ~batch_size
    ~learning_rate
    ~report:(fun t ->
      let (c,d) = report_accuracy Mnist.input_size vd t in
      Printf.printf "%d out of %d\n%!" c d)
    rmse_cdf
    t;
  t

