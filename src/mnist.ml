(* Specific to training against MNIST *)

let input_size = 28 * 28

let desc hidden_layer_size =
  Desc.(init input_size
       |> add_layer Nonlinearity.Sigmoid hidden_layer_size
       |> add_layer Nonlinearity.Softmax 10)

