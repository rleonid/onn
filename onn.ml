
open Lacaml.D
open Bigarray

type hidden_desc = nonlinearity * int
and nonlinearity =
  | Tanh
  | Sigmoid
  | ReLUs
  | Softmax
  | None
and 'a input_layer =
  | Input of 'a (*int -> int layer *)
and ('nl, 'b) hidden_layer =
  | Hidden of ('nl * 'b) (*: hidden_desc -> hidden_desc layer*)
and ('a, 'nl, 'b) network = 
  { input       : 'a input_layer
  ; hidden      : ('nl, 'b) hidden_layer array
  ; output_size : int
  }

let example     = 
  { input       = Input 2
  ; hidden      = [| Hidden (Tanh, (100, false)); Hidden (Softmax, (2,false)) |]
  ; output_size = 2
  }

module BA = Bigarray.Array2

let load_matrix f r c =
  let m = Mat.make0 r c in
  let ic = open_in f in
  let ic_csv = Csv.of_channel ~separator:' ' ~strip:true ~has_header:false ic in
  let _rows = 
    Csv.fold_left ~f:(fun r str_lst ->
      let _cols = List.fold_left (fun c str ->
        let () = BA.unsafe_set m r c (float_of_string str) in
        c + 1) 1 str_lst
      in
      r + 1) ~init:1 ic_csv 
  in
  let () = close_in ic in
  m

let apply_nl = function
  | Tanh      -> fun v -> Vec.map tanh v
  | Sigmoid   -> fun v -> Vec.map (fun x -> 1. /. (1. +. exp (-.x))) v
  | ReLUs     -> fun v -> Vec.map (max 0.) v
  | Softmax   -> fun v ->
                    let e = Vec.exp v in
                    let d = Vec.sum e in
                    Vec.map (fun e_i -> e_i /. d) e
  | None      -> fun v -> v

let deriv_nl = function
  | Tanh      -> fun v -> Vec.sub (Vec.make (Vec.dim v) 1.) 
                            (Vec.map (fun x -> tanh x *. tanh x) v)
  | Softmax   -> fun v -> v
  | None      -> fun v -> v
  | Sigmoid   ->
      fun v ->
        let ev = Vec.exp v in
        let dn = (Vec.map (fun ez -> (1. +. ez) *. (1. +. ez)) ev) in
        Vec.div ev dn
  | _         -> failwith "Not implemented"
                  


let alloc_i = function
  | Input size      -> Input size, size

let vi = ref (fun n -> Vec.make0 n)

let seed_ref = ref (Array1.of_array Int32 Fortran_layout [| 0l;0l;0l;0l |])

let initialize_random_vec_gen seed =
  let seed1 = Int32.of_int seed in
  let seed2 = Int32.add seed1 1l in
  let seed3 = Int32.add seed2 1l in
  let seed4 = Int32.add (Int32.mul (Int32.add seed3 1l) 2l) 1l in
  seed_ref := Array1.of_array Int32 Fortran_layout [|seed1; seed2; seed3; seed4 |];
  let f n = larnv ~idist:`Normal ~iseed:(!seed_ref) ~n () in
  vi := f;
  f

type layer_calc =
  { weight_input  : vec
  ; activation    : vec
  ; weights       : mat
  ; constants     : [`LastColMat | `Specific of vec]
  }

let alloc_h n = function
  | Hidden (nl, (m, inside))  ->
      let f = (apply_nl nl), (deriv_nl nl) in
      let weights, constants =
        if inside then
          Mat.of_col_vecs (Array.init (n + 1) (fun _ -> !vi m)), `LastColMat
        else
          Mat.of_col_vecs (Array.init n (fun _ -> !vi m)), `Specific (!vi m)
      in
      let lc = { weight_input = !vi m
               ; activation = !vi m
               ; weights ; constants }
      in
      Hidden (f, lc), m

let alloc { input; hidden; output_size } =
  let input, i_s = alloc_i input in
  let ps     = ref i_s in
  let hidden =
    Array.map (fun hl ->
      let m, ms = alloc_h !ps hl in
      ps := ms;
      m) hidden
  in
  if !ps <> output_size then
    raise (Invalid_argument "incorrect final size")
  else
    { input; hidden; output_size }

let _ = initialize_random_vec_gen 1
let allocated = alloc example

(* a = previous layer activation. *)
let apply_hidden a = function
  | Hidden ((nl,_), l) ->
      let weight_input =
        (* An open question as to which is faster. *)
        match l.constants with
        | `Specific c -> gemv ~y:(copy ~y:l.weight_input c) l.weights a
        | `LastColMat ->
            let cols = Mat.dim2 l.weights in
            gemv ~y:(copy ~y:l.weight_input (Mat.col l.weights cols))
              ~n:(cols - 1) ~beta:1.0 l.weights a
      in
      let activation = copy ~y:l.activation (nl weight_input) in
      activation

let eval {input = Input i_v; hidden; _ } =
  (fun v_0 ->
    if Vec.dim v_0 <> i_v then
      raise (Invalid_argument "improper size.")
    else
      Array.fold_left apply_hidden v_0 hidden)

let backprop_hidden h = function
  | [] -> raise (Invalid_argument "no starting error passed to backprop")
  | ((dl,_werror) :: _tl) as elst ->
      match h with
      | Hidden ((_,dnl), l) ->
          let sz = dnl l.activation in
          let n = Vec.dim l.activation in
          let dl_l =
            match l.constants with
            | `Specific _ -> Vec.mul (gemv ~trans:`T l.weights dl) sz
            | `LastColMat -> Vec.mul (gemv ~trans:`T ~n l.weights dl) sz
          in
          let werror = ger dl_l l.activation (Mat.make0 n n) in
          (dl_l,werror) :: elst


let backprop {hidden; _ } =
    Array.fold_right backprop_hidden hidden 

let eval_array i =
  let f = eval i in
  (fun input_array -> f (Vec.of_array input_array))

(*
let xm = load_matrix "X.mtxt" 5000 400 
let ym = load_matrix "y.mtxt" 5000 1
let theta1 = load_matrix "Theta1.mtxt" 25 401
let theta2 = load_matrix "Theta2.mtxt" 10 26

let example2 = 
  { input   = Input 400
  ; hidden  = [| Hidden ((apply_nl Sigmoid, deriv_nl Sigmoid),
                  {work_space = !vi 25; weights = theta1; constants = `LastColMat})
               ; Hidden ((apply_nl Sigmoid, deriv_nl Sigmoid),
                  {work_space = !vi 10; weights = theta2; constants = `LastColMat})
              |]
  ; output_size  = 10
  }
  *)
 
(* how would we include a perceptron?
it is a dot product of the input times weights,
with a threshold determining 0 or 1.
*)

type perceptron =
  { weights : vec
  ; bias    : float
  }

let evalp input p =
  if dot p.weights input +. p.bias <= 0.0 then
    0.0
  else
    1.0

let shuffle_by_rows m =
  for n = Array.length m - 1 downto 1 do
    let k = Random.int (n + 1) in
    if k <> n then
      let buf = m.(n) in
      m.(n) <- m.(k);
      m.(k) <- buf
  done;
  m

let shuffle_gen n =
  let arr = Array.init n (fun i -> i) in
  for i = 0 to n - 2 do
    arr.(i) <- arr.(i) + Random.int (n - i)
  done;
  arr

let permutation n =
  shuffle_gen n
  |> Array.map (fun i -> Int32.of_int (i + 1))
  |> Array1.of_array Int32 Fortran_layout 
 

let rmse_cost y y_hat = (Vec.ssqr_diff y y_hat) /. 2.
let rmse_cdf y y_hat = Vec.sub y y_hat

(* TODO; This method really highlights,
  how it would be more efficient to permute columns,
  and therefore orient our training data (and label) as
  columns. *)
let train_on training_offset td learning_rate t (cost,cdf) =
  failwith "Not implemented"
  (*
  let data_size = Mat.dim1 td - 1 in
  let ws = Vec.make0 (data_size - training_offset) in
  let e = eval t in
  let deltas =
    Mat.fold_cols (fun acc example ->
      let training = copy ~y:ws ~n:training_offset example in
      let y_hat   = e training in
      let y       = copy ~ofsx:(data_size + 1) example in
      let _c      = cost y y_hat in
      let costd   = cdf y y_hat in
      let deltas  = backprop learning_rate t [costd] in
      deltas :: acc) [] td
  in
  (* TODO: fold this into the previous fold, need a 'Running' for a vector. *)
  let collapsed_delta =
    match deltas with
    | []      -> raise (Invalid_argument "no deltas ?")
    | [one]   -> one
    | h :: t  ->
        List.fold_left
          (fun (dv,wv) (d,w) -> Vec.sum dv d, Vec.sum wv w)
          (Vec.make0 
          deltas

  *)

 
let sgd_epoch training_offset td batch_size learning_rate t c =
  let td_size = Mat.dim2 td in
  let perm = permutation td_size in
  let () = lapmt td perm in
  let num_bat = td_size / batch_size in
  for i = 0 to num_bat do
    (* Is this copy faster than Mat.of_cols (Mat.copy ...)? *)
    let epoch_td = lacpy ~ac:(i * batch_size + 1) ~n:batch_size td in
    train_on training_offset epoch_td learning_rate t c
  done

let sgd training_offset training_data ~epochs ~batch_size learning_rate 
  ?(report=(fun _ -> ())) t c =
  for i = 1 to epochs do
    sgd_epoch training_offset training_data batch_size learning_rate t c;
    report t
  done

