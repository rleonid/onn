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
  let h1 = Array2.dim1 input in
  let dw = h1 - fs + 2 * zp in
  if dw mod st <> 0 then
    iaw h1
  else
    let h2 = dw / st + 1 in
    let w1 = Array2.dim2 input in
    let dh = w1 - fs + 2 * zp in
    if dh mod st <> 0 then
      iaw w1
    else
      let w2 = dh / st + 1 in
      (h2, w2)

let field_size_to_fortran_layout_indices n =
  if n mod 2 = 0 then
    ia "field_size_must be odd %d" n
  else
    Array.init (n * n) (fun i -> i mod n + 1, 1 + i / n)

let to_bounds_checked_get ?(z=0.) m =
  let nr = Array2.dim1 m in
  let nc = Array2.dim2 m in
  fun r c ->
    (* This can't be good for our health. *)
    if r < 1 then z else
      if r > nr then z else
        if c < 1 then z else
          if c > nc then z else
            Array2.unsafe_get m r c

let unroll_mat_gen ~fs ~zp ?z ~st m g i =
  (*let h,w   = conv_widths m ~fs ~zp ~st in *)
  let h     = Array2.dim1 m in
  let w     = Array2.dim2 m in
  let inds  = field_size_to_fortran_layout_indices fs in
  let bcg   = to_bounds_checked_get m ?z in
  let off ro co = fun (r, c) -> bcg (ro + r) (co + c) in
  (* this is the 'top right' offset *)
  let last_col_offset = h + zp - fs in
  let last_row_offset = w + zp - fs in
  let rec loop ro co acc =
    (*let () = printf "at r:%d c:%d\n" ro co in *)
    let nacc = g (off ro co) inds acc in
    if ro + st <= last_row_offset then
      loop (ro + st) co nacc
    else if co + st <= last_col_offset then
      loop (-zp) (co + st) nacc
    else
      nacc
  in
  loop (-zp) (-zp) i

let unroll_mat_to_lst ~fs ~zp ?z ~st m =
  unroll_mat_gen  ~fs ~zp ?z ~st m
    (fun m inds acc -> Array.map m inds :: acc)
    []
  |> List.rev

let unroll_mat_to_mat ?mat ~fs ~zp ?z ~st m =
  let ro, mat =
    match mat with
    | None        ->
        let (w,h) = conv_widths m ~fs ~zp ~st in
        0, Array2.create Float64 Fortran_layout (w * h) (fs * fs)
    | Some (r, m) ->
        r, m
  in
  let _ =
    unroll_mat_gen ~fs ~zp ?z ~st m
      (fun m inds col_idx ->
        Array.iteri (fun i ind ->
          Array2.set mat (ro + i + 1) col_idx (m ind))
          inds;
        col_idx + 1)
      1
  in
  mat

let unroll_mat_sq_to_vec m =
  let fs = Array2.dim1 m in
  let n2 = Array2.dim2 m in
  if fs <> n2 then
    ia "Supposed to be square"
  else
    match unroll_mat_to_lst m ~fs ~zp:0 ~z:nan ~st:1 with
    | [ v ] -> Array1.of_array Float64 Fortran_layout v
    | _ -> ia "Flawed logic"

let unroll_mat_array ~fs ~zp ?z ~st arr =
  let (w,h) = conv_widths arr.(0) ~fs ~zp ~st in
  let n = Array.length arr in
  let rowi = w * h in
  let rows = rowi * n in
  let cols = fs * fs in
  let mat = Array2.create Float64 Fortran_layout rows cols in
  Array.iteri (fun i m ->
    ignore(unroll_mat_to_mat ~fs ~zp ?z ~st ~mat:(i * rowi, mat) m))
      arr;
  mat

let weights_to_col warr = Mat.as_vec (Mat.of_col_vecs warr)
