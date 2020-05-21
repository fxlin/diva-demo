## features needed:


visualization

- framemap. to show fine-grained progress of cam.

    for each query, the cam keeps track the state of each frame. 
         unprocesssed, filtered, ranked, queued, sent...? 
         possible states can be simple. can be refined later
         each frame carry timestsamp? probably not
         
    to retrieve framemap:       
        ctrl polls cam. the cam returns a framemap. 
        too much data? e.g. 100K frames in a query? 
            can apply grpc compression.
            https://github.com/grpc/grpc/tree/master/examples/python/compression
             
        don't opt unless it becomes a problem
        
    server: 
        render the framemap. 
        reconstruct the overall progress (pos, neg, etc.)
        http://biobits.org/bokeh-flask.html
        
        good example:
        https://davidhamann.de/2018/02/11/integrate-bokeh-plots-in-flask-ajax/
        
        example: 
        https://github.com/realpython/flask-bokeh-example/tree/master/flask-bokeh-sample
        
        
        https://docs.bokeh.org/en/latest/docs/user_guide/server.html
        https://hplgit.github.io/web4sciapps/doc/pub/._web4sa_flask013.html
        https://docs.bokeh.org/en/latest/docs/gallery.html#gallery
                 
- progress curve. 
    js? bootstrap console framework? 
        
    or server-side rendering (python?)
        
     
- multipass ranking 
    on cam. need to choose the operators. what are they?    