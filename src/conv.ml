(* Represent convolutional multiplications. *)

open Bigarray
open Printf

let ia fmt = ksprintf invalid_arg fmt

(* fs = field size
   zp = zero padding
   st = stride
 *)
let conv_widths input ~fs ~zp ~st =
  let iaw w = 
    ia "Can't fit stride %d in width %d, field size %d and padding %d"
      st w fs zp
  in
  let w1 = Array2.dim1 input in
  let dw = w1 - fs + 2 * zp in
  if dw mod st <> 0 then
    iaw w1
  else
    let w2 = dw + 1 in
    let h1 = Array2.dim2 input in
    let dh = h1 - fs + 2 * zp in
    if dh mod st <> 0 then
      iaw h1
    else
      let h2 = dh + 1 in
      (w2, h2)

let field_size_to_fortran_layout_indices n =
  if n mod 2 = 0 then
    ia "field_size_must be odd %d" n
  else
    Array.init (n * n) (fun i -> 1 + i / n, i mod n + 1) 

let to_bounds_checked_get m ~z =
  let nr = Array2.dim1 m in
  let nc = Array2.dim2 m in
  fun r c ->
    (* This can't be good for our health. *)
    if r < 1 then z else
      if r > nr then z else
        if c < 1 then z else
          if c > nc then z else
            Array2.unsafe_get m r c
 
let unroll_mat m ~fs ~zp ~z ~st =
  let (w,h) = conv_widths m ~fs ~zp ~st in
  let inds  = field_size_to_fortran_layout_indices fs in
  let bcg = to_bounds_checked_get m ~z in
  let off ro co = fun (r, c) -> bcg (ro + r) (co + c) in
  let rec loop ro co acc =
    let () = printf "at %d %d\n" ro co in
    let column = Array.map (off ro co) inds in
    let nacc = column :: acc in
    if co + st < h - zp then      (* this is the 'top right' offset *)
      loop ro (co + st) nacc
    else if ro + st < w - zp then
      loop (ro + st) (-zp) nacc
    else
      List.rev nacc 
  in
  loop (-zp) (-zp) []
