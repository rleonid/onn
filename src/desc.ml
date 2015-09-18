
(****** Layer Description ******)

type desc =
  | Input   of int
  | Hidden  of Nonlinearity.t * int * desc

let init n = Input n
let add_layer nl n p = Hidden (nl, n, p)

let example =
  let open Nonlinearity in
  init 10
  |> add_layer Tanh 100
  |> add_layer Softmax 2


let to_size = function | Input n | Hidden (_, n, _) -> n
