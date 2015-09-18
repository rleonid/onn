
open Lacaml.D

type t =
  | Tanh
  | Sigmoid
  | ReLUs
  | Softmax

(** Applying nonlinearities *)
let apply = function
  | Tanh      -> fun v -> Vec.map tanh v
  | Sigmoid   -> fun v -> Vec.map (fun x -> 1. /. (1. +. exp (-.x))) v
  | ReLUs     -> fun v -> Vec.map (max 0.) v
  | Softmax   -> fun v ->
                    let e = Vec.exp v in
                    let d = Vec.sum e in
                    Vec.map (fun e_i -> e_i /. d) e

let deriv = function
  | Tanh      -> fun v -> Vec.sub (Vec.make (Vec.dim v) 1.)
                            (Vec.map (fun x -> tanh x *. tanh x) v)
  | Softmax   -> fun v -> v
  | Sigmoid   ->
      fun v ->
        let ev = Vec.exp v in
        let dn = (Vec.map (fun ez -> (1. +. ez) *. (1. +. ez)) ev) in
        Vec.div ev dn
  | _         -> failwith "Not implemented"

