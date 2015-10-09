
let get_color_array values_array r =
  aligned_native (fun i -> values_array.(r).(i)) Graphics.rgb

let draw_and_inspect ?(cg=true) ?(x=100) ?(z=1) ?(y=100) ?label ca =
  if cg then Graphics.clear_graph ();
  let ca =
    if z <= 1 then ca else
      let n = Array.length ca in
      let m = Array.length ca.(0) in
      Array.init (n * z) (fun i ->
        Array.init (m * z) (fun j ->
          ca.(i / z).(j / z)))
  in
  let im = Graphics.make_image ca in
  Graphics.draw_image im x y;
  match label with | None -> ()
  | Some l ->
    begin
      Graphics.moveto x (y - 10);
      Graphics.draw_string l
    end
