(* Specific to training against MNIST *)

let input_size = 28 * 28

let desc num_hidden_nodes =
  Desc.(init input_size
       |> add_layer Nonlinearity.Sigmoid num_hidden_nodes
       |> add_layer Nonlinearity.Softmax 10)

