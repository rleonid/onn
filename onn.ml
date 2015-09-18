
open Lacaml.D
open Bigarray

(****** Layer Description ******)

(*type hidden_desc = nonlinearity * int*)
type nonlinearity =
  | Tanh
  | Sigmoid
  | ReLUs
  | Softmax

type desc =
  | Input   of int
  | Hidden  of nonlinearity * int * desc

let init n = Input n
let add_layer nl n p = Hidden (nl, n, p)

let example =
  init 10
  |> add_layer Tanh 100
  |> add_layer Softmax 2

let apply_nl = function
  | Tanh      -> fun v -> Vec.map tanh v
  | Sigmoid   -> fun v -> Vec.map (fun x -> 1. /. (1. +. exp (-.x))) v
  | ReLUs     -> fun v -> Vec.map (max 0.) v
  | Softmax   -> fun v ->
                    let e = Vec.exp v in
                    let d = Vec.sum e in
                    Vec.map (fun e_i -> e_i /. d) e

let deriv_nl = function
  | Tanh      -> fun v -> Vec.sub (Vec.make (Vec.dim v) 1.)
                            (Vec.map (fun x -> tanh x *. tanh x) v)
  | Softmax   -> fun v -> v
  | Sigmoid   ->
      fun v ->
        let ev = Vec.exp v in
        let dn = (Vec.map (fun ez -> (1. +. ez) *. (1. +. ez)) ev) in
        Vec.div ev dn
  | _         -> failwith "Not implemented"


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
  { nonlinearity : nonlinearity
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

let to_size = function | Input n | Hidden (_, n, _) -> n

let rec repr (acc, l) = function
  | Input n             -> n, acc, l
  | Hidden (nl, n, pl)  ->
      let b = !vi n in
      let m = to_size pl in
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
  |> apply_nl hl.nonlinearity

let eval t v_0 =
  if Vec.dim v_0 <> t.input_size then
    raise (Invalid_argument "improper size.")
  else
    Array.fold_left apply v_0 t.hidden_layers

let backprop_l iteration hl prev_error =
  let i_f = float iteration in
  let ifi = 1. /. i_f in
  let sz = (deriv_nl hl.nonlinearity) hl.weight_input in
  let dl = Vec.mul prev_error sz in
  let () = (* update bias error *)
    scal (1. -. ifi) hl.bias_e;
    axpy ~alpha:ifi dl hl.bias_e
  in
  let dl_n = gemv ~trans:`T hl.weights dl in
  let () = (* update weight error *)
    Mat.scal (1. -. ifi) hl.weights_e;
    ignore (ger ~alpha:ifi dl hl.p_activation hl.weights_e)
  in
  dl_n

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
    (*let () = Format.print_flush () in
    let () = pp_vec Format.std_formatter hl.bias_e in
    let () = Printf.printf "----------------\n%!" in *)
    axpy ~alpha hl.bias_e hl.bias;
    Mat.axpy ~alpha hl.weights_e hl.weights)
    t.hidden_layers

let train_on training_offset td learning_rate cdf t =
  let data_size = Mat.dim1 td - 1 in
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

(***** Load MNIST data *****)
#use "mnist/load.ml" ;;

test_images_fname := Filename.concat "mnist" !test_images_fname ;;
test_labels_fname := Filename.concat "mnist" !test_labels_fname ;;
train_images_fname := Filename.concat "mnist" !train_images_fname ;;
train_labels_fname := Filename.concat "mnist" !train_labels_fname ;;

let data = function
  | `Test  -> join (parse_images !test_images_fname) (parse_labels !test_labels_fname)
  | `Train -> join (parse_images !train_images_fname) (parse_labels !train_labels_fname)

let split_validation ?seed size td =
  let () = match seed with | None -> () | Some s -> Random.init s in
  let n  = Mat.dim2 td in
  let permutation = permutation_gen n in
  lapmt td permutation;
  let t = lacpy ~n:(n - size) td in
  let v = lacpy ~ac:(n - size + 1) td in
  t, v

let mnistnn_desc hidden_layer_size =
  init (28 * 28)
  |> add_layer Sigmoid hidden_layer_size
  |> add_layer Sigmoid 10

let mnist_nn hls = compile (mnistnn_desc hls)

let max_idx_in m =
  let _, mx, _ =
    Vec.fold (fun (i,j,m) v -> if v > m then (i+1,i,v) else (i+1,j,m))
      (1, 0, neg_infinity) m
  in
  mx

let correct y_p y =
  let i = max_idx_in y_p in
  Array1.get y i = 1.0

let report_accuracy training_offset d =
  let ws = Vec.make0 training_offset in
  let m = Mat.dim2 d in
  (fun t ->
    let e = eval t in
    let correct =
      Mat.fold_cols (fun a c ->
        let training = copy ~y:ws ~n:training_offset c in
        let y_hat    = e training in
        let y        = copy ~ofsx:(training_offset + 1) c in
        if correct y_hat y then a + 1 else a) 0 d
    in
    (correct, m))

let td_vd_ref = ref None

let do_it ~batch_size ~hidden_layers ~epochs ~learning_rate =
  let td, vd =
    match !td_vd_ref with
    | None ->
      let d = data `Train in
      let s = split_validation 10000 d in
      td_vd_ref := Some s;
      s
    | Some p -> p
  in
  let toffset = 28 * 28 in
  let t = mnist_nn hidden_layers in
  (*let cost = rmse_cdf, rmse_cost in *)
  sgd toffset td
    ~epochs
    ~batch_size
    ~learning_rate
    ~report:(fun t ->
      let (c,d) = report_accuracy toffset vd t in
      Printf.printf "%d out of %d\n%!" c d)
    rmse_cdf
    t;
  t

let simple_t () = do_it ~batch_size:50 ~hidden_layers:20 ~epochs:30 ~learning_rate:0.3 ;;
