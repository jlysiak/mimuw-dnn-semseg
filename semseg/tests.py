
    def test_pred_pipe(self, 
            out_dir="pred_pipe_test", 
            train_outs="training", 
            valid_outs="validation"):

        conf = CONF(self.config)
        log = lambda x: self.log(x)
        
        v_outs = os.path.join(out_dir, valid_outs)
        if not os.path.exists(v_outs):
            os.makedirs(v_outs)

        log('******* Prediction input pipe test')
        it_p = self.setup_pipe(False)
        next_p = it_p.get_next()
        with tf.Session() as sess:
            sess.run([it_p.initializer])
            try:
                i = 0
                while True:
                    img, lab, name, shape, box = sess.run(next_p)
                    result = Image.fromarray(img.astype(np.uint8))
                    result.save(os.path.join(v_outs, "%d.jpg" % i))
                    i += 1
            except tf.errors.OutOfRangeError:
                print("End...")


    def test_train_pipe(self, 
            out_dir="train_pipe_test", 
            train_outs="training", 
            valid_outs="validation"):
        """
        Handy method to check input pipe.
        """
        conf = CONF(self.config)
        log = lambda x: self.log(x)
        
        t_outs = os.path.join(out_dir, train_outs)
        v_outs = os.path.join(out_dir, valid_outs)
        if not os.path.exists(t_outs):
            os.makedirs(t_outs)
        if not os.path.exists(v_outs):
            os.makedirs(v_outs)

        log('******* Input pipe test')
        it_t, it_v, v_trans_num = self.setup_pipe()
        next_t = it_t.get_next()
        next_v = it_v.get_next()
         
        with tf.Session() as sess:
            sess.run([it_t.initializer, it_v.initializer])
            try:
                i = 0
                while True:
                    a, b = sess.run(next_t)
                    print(i, a.shape, b.shape)
                    for k in range(len(a)):
                        result = Image.fromarray(a[k].astype(np.uint8))
                        result.save(os.path.join(t_outs, "%d-a.jpg" % i))
                        result = Image.fromarray(b[k].astype(np.uint8).reshape(256,256), mode="L")
                        result.save(os.path.join(t_outs, "%d-b.png" % i))
                        i += 1
            except tf.errors.OutOfRangeError:
                print("End...")

            try:
                i = 0
                while True:
                    a, b = sess.run(next_v)
                    imgn = i // v_trans_num
                    transn = i % v_trans_num
                    print(i, imgn, transn, a.shape, b.shape)
                    result = Image.fromarray(a.astype(np.uint8))
                    result.save(os.path.join(v_outs, "%d-%d-a.jpg" % (imgn, transn)))
                    result = Image.fromarray(b.astype(np.uint8).reshape(256,256), mode="L")
                    result.save(os.path.join(v_outs, "%d-%d-b.png" % (imgn, transn)))
                    i += 1
            except tf.errors.OutOfRangeError:
                print("End...")


