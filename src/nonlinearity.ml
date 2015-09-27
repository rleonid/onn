
open Lacaml.D

type t =
  | Tanh
  | Sigmoid
  | ReLUs
  | Softmax

let to_softmax v =
  let e = Vec.exp v in
  let d = Vec.sum e in
  Vec.map (fun e_i -> e_i /. d) e



(** Applying nonlinearities *)
let apply = function
  | Tanh      -> fun v -> Vec.map tanh v
  | Sigmoid   -> fun v -> Vec.map (fun x -> 1. /. (1. +. exp (-.x))) v
  | ReLUs     -> fun v -> Vec.map (max 0.) v
  | Softmax   -> to_softmax

type jacobian =
  | Diagonal of vec
  | Matrix of mat

let deriv = function
  | Tanh      -> fun v ->
      Diagonal (Vec.sub (Vec.make (Vec.dim v) 1.)
                        (Vec.map (fun x -> tanh x *. tanh x) v))
  | Softmax   ->
      fun v ->
        (* See http://stats.stackexchange.com/questions/79454/softmax-layer-in-a-neural-network,
        for reference, but the deriv aint that hard *)
        let s = to_softmax v in
        Matrix (ger ~alpha:(-1.) s s (Mat.of_diag s))
  | Sigmoid   ->
      fun v ->
        let ev = Vec.exp v in
        let dn = (Vec.map (fun ez -> (1. +. ez) *. (1. +. ez)) ev) in
        Diagonal (Vec.div ev dn)
  | _         -> failwith "Not implemented"

