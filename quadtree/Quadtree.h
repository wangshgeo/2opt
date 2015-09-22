#ifndef QUADTREE_H_
#define QUADTREE_H_


//Generate the quadtree.
  dtype xmin,xmax,ymin,ymax;
  reduce(x,n,&xmin,&xmax);
  reduce(y,n,&ymin,&ymax);
  morton_key_type* point_morton_pairs = new morton_key_type[n];
  make_morton_keys_serial(point_morton_pairs, x, y, n, xmin, xmax, ymin, ymax);
  Node* tree = construct_quadtree_serial(
    NULL, // We are not inputing a head, since we are retrieving the head.
    -1, // The head is not a child, so its child index is -1.
    point_morton_pairs,
    n, 
    0, // Current level of root is 0.
    x, 
    y
  );

  // Allocate and initialize data structures.
  cost_t* segments = new cost_t[n];
  compute_segment_lengths(x, y, n, segments);
  dtype* segment_center_x = new dtype[n];
  dtype* segment_center_y = new dtype[n];
  compute_segment_centers(segment_center_x,segment_center_y,x,y,n);
  mtype* point_morton_keys = new mtype[n];
  ordered_point_morton_keys(point_morton_pairs, point_morton_keys, n);

  // Populate the quadtree with segments.
  insert_segments(tree, point_morton_keys, n);
  
  // Tree checks.
  // print_quadtree(tree, "");
  // write_tour(x,y,n);












    int i_best_quadtree=0,j_best_quadtree=0;
    cost_t cost_original;
    cost_t cost_quadtree;
    Node* best_node;
    int best_segment_index;//index in best_node segments (container).

    timer.start();
    best_improvement_quadtree( &i_best_quadtree, &j_best_quadtree, 
      &cost_quadtree, &best_node, &best_segment_index,
      segments, segment_center_x, segment_center_y,
      x, y, n, tree, map );
    fprintf(stdout, "\tQuadtree best: %d %d %f\n", i_best_quadtree, j_best_quadtree, (dtype) cost_quadtree);
    quadtree_best_improvement_time += timer.stop();

    timer.start();
    swap(i_best_quadtree, j_best_quadtree, x, y);
    flip<int>(map,i_best_quadtree,j_best_quadtree);
    flip_time += timer.stop();






  delete[] x;
  delete[] y;
  delete[] point_morton_pairs;
  delete[] segments;
  delete[] segment_center_x;
  delete[] segment_center_y;
  delete[] map;
  delete[] point_morton_keys;
  destroy_quadtree_serial(tree);




#endif