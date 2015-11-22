
let m1 =
  Array.map (Array.map float)
    [| [| 0;1;2;2;2 |]
     ; [| 1;2;1;0;1 |]
     ; [| 2;0;2;1;1 |]
     ; [| 1;1;0;0;2 |]
     ; [| 2;1;2;1;1 |]
     |]
  |> Mat.of_array

let m2 =
  Array.map (Array.map float)
    [| [| 1;1;2;1;1 |]
     ; [| 2;2;0;0;0 |]
     ; [| 1;0;1;1;1 |]
     ; [| 0;1;0;2;1 |]
     ; [| 0;2;0;1;1 |]
     |]
  |> Mat.of_array

let m3 = 
  Array.map (Array.map float)
    [| [| 1;0;2;1;1 |]
     ; [| 2;2;2;0;0 |]
     ; [| 0;2;1;2;1 |]
     ; [| 0;2;0;0;2 |]
     ; [| 0;0;1;0;2 |]
     |]
  |> Mat.of_array

let w00 =
  Array.map (Array.map float)
    [| [| 0;0;0 |]
     ; [| 1;0;0 |]
     ; [| 1;1;0 |]
     |]
  |> Mat.of_array
  |> unroll_mat_sq_to_vec

let w01 =
  Array.map (Array.map float)
    [| [| 1;-1;1 |]
     ; [| 0;1;-1 |]
     ; [| -1;0;1 |]
     |]
  |> Mat.of_array
  |> unroll_mat_sq_to_vec

let w02 =
  Array.map (Array.map float)
    [| [| -1;1;1 |]
     ; [| 0;-1;1 |]
     ; [| 1;-1;0 |]
     |]
  |> Mat.of_array
  |> unroll_mat_sq_to_vec


