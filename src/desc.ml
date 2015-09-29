
(****** Layer Description ******)

type desc =
  | Input of int
  | Hidden of Nonlinearity.t * int * desc
  | ScaledInit of Nonlinearity.t * int * desc

let init n = Input n
let add_hidden nl n p = Hidden (nl, n, p)
let add_scaled_init nl n p = ScaledInit (nl, n, p)

let example =
  let open Nonlinearity in
  init 10
  |> add_hidden Tanh 100
  |> add_hidden Softmax 2


let to_size = function
  | Input n
  | Hidden (_, n, _)
  | ScaledInit (_, n, _) -> n
