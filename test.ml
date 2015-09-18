
let () =
  let t = Onn.do_it ~batch_size:40 ~hidden_layers:100 ~epochs:30 ~learning_rate:0.5 in
  let test_d = Onn.data `Test in
  let (c,t) = Onn.report_accuracy (28 * 28) test_d t in
  Printf.printf "Got %d out of %d on test data." c t
