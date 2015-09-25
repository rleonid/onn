
open Lacaml.D
open Bigarray

let invalidArg fmt = Printf.ksprintf (fun s -> raise (Invalid_argument s)) fmt

module Array = struct
  include Array

  let fold_left2 f i a b =
    let n = Array.length a in
    let m = Array.length b in
    if n <> m then
      invalidArg "fold_left2: unequal lengths %d and %d" n m
    else
      begin
        let r = ref i in
        for i = 0 to n - 1 do
          r := f !r a.(i) b.(i)
        done;
        !r
      end

  let fold_right2 f a b i =
    let n = Array.length a in
    let m = Array.length b in
    if n <> m then
      invalidArg "fold_right2: unequal lengths %d and %d" n m
    else
      begin
        let r = ref i in
        for i = n - 1 downto 0 do
          r := f a.(i) b.(i) !r
        done;
        !r
      end

end

module Mat = struct
  include Mat

  (** This is mostly used in the application of the nonlinearity,
      it probably makes more sense to have specialized routines on
      a per nonlinearity to perform mat -> mat mapping. *)
  let map_cols f m =
    Mat.fold_cols (fun a vec -> f vec :: a) [] m
    |> List.rev
    |> Array.of_list
    |> Mat.of_col_vecs

end

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
  ; bias         : vec
  ; weights      : mat
  ; bias_e       : vec
  ; weights_e    : mat
  }

let copy_hidden h =
  { h with bias      = copy h.bias
         ; weights   = lacpy h.weights
         ; bias_e    = copy h.bias_e
         ; weights_e = lacpy h.weights_e
         }

type 'a ed =
  { weight_input : 'a
  ; p_activation : 'a
  }

let hidden ?batch nonlinearity weights bias =
  let m = Mat.dim1 weights in
  let n = Mat.dim2 weights in
  { nonlinearity
  ; bias
  ; weights
  ; bias_e       = Vec.make0 m
  ; weights_e    = Mat.make0 m n
  }

let single_ed ~m ~n =
  { weight_input = Vec.make0 m
  ; p_activation = Vec.make0 n
  }

let batch_ed ~m ~n b =
  { weight_input = Mat.make0 m b
  ; p_activation = Mat.make0 n b
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

let copy_t t =
  { t with hidden_layers = Array.map copy_hidden t.hidden_layers }

let compile desc =
  let input_size, as_lst, len = repr ([],0) desc in
  let hidden_layers = Array.of_list as_lst in
  let output_size   = Vec.dim hidden_layers.(len - 1).bias in
  { output_size; hidden_layers; input_size}

let single_eda t =
  t.hidden_layers
  |> Array.map (fun hl ->
      let m = Mat.dim1 hl.weights in
      let n = Mat.dim2 hl.weights in
      single_ed ~m ~n)

let batch_eda b t =
  t.hidden_layers
  |> Array.map (fun hl ->
      let m = Mat.dim1 hl.weights in
      let n = Mat.dim2 hl.weights in
      batch_ed ~m ~n b)

(* pla -> previous layers activation *)
let apply pla hl ed =
  ignore (copy ~y:ed.p_activation pla);
  ignore (copy ~y:ed.weight_input hl.bias);
  gemv ~beta:1. ~y:ed.weight_input hl.weights pla
  |> Nonlinearity.apply hl.nonlinearity

let apply_m plam hl ed =
  let m = Mat.dim2 plam in
  (* Keep the previous layers activation for backprop *)
  ignore (lacpy ~b:ed.p_activation plam);
  (* Initialize the weight projection with bias. *)
  let bias_mat = Mat.of_col_vecs (Array.init m (fun _ -> hl.bias)) in
  ignore (lacpy ~b:ed.weight_input bias_mat);
  gemm ~beta:1. ~c:ed.weight_input hl.weights plam
  |> Mat.map_cols (Nonlinearity.apply hl.nonlinearity)

let eval t eda v_0 =
  if Vec.dim v_0 <> t.input_size then
    raise (Invalid_argument "improper size.")
  else
    Array.fold_left2 apply v_0 t.hidden_layers eda

let eval_m t eda vm =
  if Mat.dim1 vm <> t.input_size then
    raise (Invalid_argument "improper size.")
  else
    Array.fold_left2 apply_m vm t.hidden_layers eda

let backprop_l iteration hl ed prev_error =
  let i_f = float iteration in
  let ifi = 1. /. i_f in
  let sz = (Nonlinearity.deriv hl.nonlinearity) ed.weight_input in
  let dl = Vec.mul prev_error sz in
  let () = (* update bias error *)
    scal (1. -. ifi) hl.bias_e;
    axpy ~alpha:ifi dl hl.bias_e
  in
  let () = (* update weight error *)
    Mat.scal (1. -. ifi) hl.weights_e;
    ignore (ger ~alpha:ifi dl ed.p_activation hl.weights_e)
  in
  gemv ~trans:`T hl.weights dl

(* each column of prev_error is an error of a training example. *)
let backprop_m hl ed prev_errors =
  let m = Mat.dim1 prev_errors in
  let n = Mat.dim2 prev_errors in
  let szs = Mat.map_cols (Nonlinearity.deriv hl.nonlinearity) ed.weight_input in
  let dls = Mat.init_cols m n (fun r c -> prev_errors.{r,c} *. szs.{r,c}) in
  let alpha = 1. /. (float n) in
  (* update bias error 
  beta is zero by default to zero out the previous error. *)
  let () = ignore (gemv ~y:hl.bias_e ~alpha dls (Vec.make n 1.)) in
  (* update weight error *)
  (* ~alpha: averaging over size of batch *)
  let () = ignore (gemm ~alpha ~transb:`T dls ed.p_activation ~c:hl.weights_e) in
  gemm ~transa:`T hl.weights dls

(* iteration: which training datum (in mini batch) are we updating
  to allow implace, (online) averaging of errors. *)
let backprop_i iteration cost_error t eda =
  let bp = backprop_l iteration in
  Array.fold_right2 bp t.hidden_layers eda cost_error

let rmse_cost y y_hat = (Vec.ssqr_diff y y_hat) /. 2.
(* This can be a source of confusion *)
type explicit_cost =  y:vec -> y_hat:vec -> vec

let rmse_cdf ~y ~y_hat = Vec.sub y_hat y

(* The 'errors' are already averaged! *)
let assign_errors learning_rate t =
  let alpha = -1.0 *. learning_rate in
  Array.iter (fun hl ->
    axpy ~alpha hl.bias_e hl.bias;
    Mat.axpy ~alpha hl.weights_e hl.weights)
    t.hidden_layers

let iterative_train training_offset td cdf t =
  let ws = Vec.make0 training_offset in
  let ed = single_eda t in
  let e = eval t ed in
  let i = ref 1 in
  Mat.map_cols (fun example ->
    let y_hat   = e (copy ~y:ws ~n:training_offset example) in
    let y       = copy ~ofsx:(training_offset + 1) example in
    let costd   = cdf ~y ~y_hat in
    let final_d = backprop_i !i costd t ed in
    incr i;
    final_d) td

let batch_train training_offset td cdf t =
  let b_size = Mat.dim2 td in
  let eda    = batch_eda b_size t in
  let y_hats = eval_m t eda (lacpy ~m:training_offset td) in
  let ys     = lacpy ~ar:(training_offset + 1) td in
  let costs  =
    Array.init b_size (fun i ->
      let j     = i + 1 in
      let y_hat = Mat.col y_hats j in
      let y     = Mat.col ys j in
      cdf ~y ~y_hat)
    |> Mat.of_col_vecs
  in
  Array.fold_right2 backprop_m t.hidden_layers eda costs

let sgd_epoch iterative training_offset td ~batch_size learning_rate c t =
  let td_size = Mat.dim2 td in
  let perm = permutation_gen td_size in
  let () = lapmt td perm in
  let num_bat = td_size / batch_size - 1 in
  for i = 0 to num_bat do
    (* Is this copy faster than Mat.of_cols (Mat.copy ...)? *)
    let epoch_td = lacpy ~ac:(i * batch_size + 1) ~n:batch_size td in
    let _final_d =
      if iterative then
        iterative_train training_offset epoch_td c t
      else
        batch_train training_offset epoch_td c t
    in
    assign_errors learning_rate t
  done

let sgd iterative training_offset training_data ~epochs ~batch_size ~learning_rate
  ?(report=(fun _ -> ())) c t =
  for i = 1 to epochs do
    sgd_epoch iterative training_offset training_data ~batch_size learning_rate c t;
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

let correct y_p y =
  let i = max_idx_in y_p in
  Array1.get y i = 1.0

let report_accuracy training_offset d =
  let ws = Vec.make0 training_offset in
  let m = Mat.dim2 d in
  (fun t ->
    let eda = single_eda t in (* TODO: need an eval mode where we don't save this. *)
    let e = eval t eda in
    let nc =
      Mat.fold_cols (fun a c ->
        let training = copy ~y:ws ~n:training_offset c in
        let y_hat    = e training in
        let y        = copy ~ofsx:(training_offset + 1) c in
        if correct y_hat y then a + 1 else a) 0 d
    in
    (nc, m))

let td_vd_ref = ref None

let load_and_save_mnist_data ?cache () =
  let d = Load_mnist.data ?cache `Train in
  let s = split_validation 10000 d in
  td_vd_ref := Some s;
  s

let do_it ?cache ~iterative ~batch_size ~hidden_layers ~epochs ~learning_rate =
  let td, vd =
    match !td_vd_ref with
    | None   -> load_and_save_mnist_data ?cache ()
    | Some p -> p
  in
  let t = compile (Mnist.desc hidden_layers) in
  sgd iterative
    Mnist.input_size td
    ~epochs
    ~batch_size
    ~learning_rate
    ~report:(fun t ->
      let (c,d) = report_accuracy Mnist.input_size vd t in
      Printf.printf "%d out of %d\n%!" c d)
    rmse_cdf
    t;
  t

