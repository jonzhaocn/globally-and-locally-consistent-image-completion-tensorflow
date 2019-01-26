import tensorflow as tf
import os
"""this file construct network models and create loss function"""


class GloballyAndLocallyConsistentImageCompletion:
    def __init__(self, img_height, img_width, local_area_shape, ckpt_dir, predicting_mode=False, batch_size=64):
        """
        init
        :param img_height: a int
        :param img_width: a int
        :param local_area_shape: a list which has a 1*2 shape, for example: (64, 64)
        :param ckpt_dir: a str point out the folder for saving model
        :param predicting_mode: bool, if True: training, if False: load trained model to complete images
        :param batch_size: setting batch size for training network
        """
        self._img_height = img_height
        self._img_width = img_width
        self._local_area_shape = local_area_shape
        self._gray_image_value = 0.437
        self._learning_rate_comp = 2e-4
        self._learning_rate_disc = 1e-5
        "adjust alpha to balance completion network and GAN"
        self._alpha = 0.0001
        self._ckpt_dir = ckpt_dir
        self._summary_log_dir = './summary'
        self._model_name = "model.ckpt"
        self._predicting_mode = predicting_mode
        self._batch_size = batch_size
        "session"
        self._sess = tf.Session()

        if self._predicting_mode is False:
            "-----training---"
            "build graph for training"
            self._ground_truth_ph = tf.placeholder(tf.float32, (self._batch_size, self._img_height, self._img_width, 3))
            self._mask_c_ph = tf.placeholder(tf.float32, (self._batch_size, self._img_height, self._img_width, 3))
            self._local_area_top_left_c_ph = tf.placeholder(tf.int32, 2)
            self._local_area_top_left_d_ph = tf.placeholder(tf.int32, 2)
            self._build_graph_for_training()
            if not os.path.exists(self._summary_log_dir):
                os.mkdir(self._summary_log_dir)
            self.summary_writer = tf.summary.FileWriter(self._summary_log_dir, self._sess.graph)
        else:
            "build graph for predicting"
            self._ground_truth_ph = tf.placeholder(tf.float32, (None, self._img_height, self._img_width, 3))
            self._mask_c_ph = tf.placeholder(tf.float32, (None, self._img_height, self._img_width, 3))
            self._build_graph_for_predicting()

        if self._ckpt_dir and os.path.exists(self._ckpt_dir):
            saver = tf.train.Saver()
            lasted_checkpoint = tf.train.latest_checkpoint(self._ckpt_dir)
            if lasted_checkpoint is not None:
                saver.restore(self._sess, lasted_checkpoint)
                print('load model:', lasted_checkpoint)
            else:
                if self._predicting_mode:
                    raise Exception('predicting mode: can not find trained model in ckpt_dir')
                else:
                    print('init global variables')
                    self._sess.run(tf.global_variables_initializer())
        else:
            if self._predicting_mode:
                raise Exception('predicting mode: ckpt_dir should not be None or ckpt_dir does not exist')
            else:
                print('init global variables')
                self._sess.run(tf.global_variables_initializer())

    def train_completion_network(self, images, mask_c):
        """
        train completion network
        :param images: 4-D array or tensor
        :param mask_c: 4-D array or tensor, has same shape of images
        :return:
        """
        feed_dict = {self._ground_truth_ph: images, self._mask_c_ph: mask_c}
        global_step = self._global_step.eval(self._sess)
        if global_step % 1000 == 0:
            _, summary = self._sess.run([self._mse_loss_optim, self._completed_images_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, global_step=global_step)
        if global_step % 100 == 0:
            _, summary = self._sess.run([self._mse_loss_optim, self._mse_loss_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, global_step=global_step)
        else:
            self._sess.run(self._mse_loss_optim, feed_dict=feed_dict)

    def train_discriminator(self, images, mask_c, local_area_top_left_c, local_area_top_left_d):
        """

        :param images: 4-D array or tensor
        :param mask_c: 4-D array or tensor, has same shape of images
        :param local_area_top_left_c: 1*2 vector, top left point position of local area of mask_c in images
        :param local_area_top_left_d: 1*2 vector, top left point position of local area of mask_d in images
        :return:
        """
        global_step = self._global_step.eval(self._sess)
        feed_dict = {self._ground_truth_ph: images, self._mask_c_ph: mask_c, self._local_area_top_left_c_ph: local_area_top_left_c,
                     self._local_area_top_left_d_ph: local_area_top_left_d}
        if global_step % 1000 == 0:
            _, summary = self._sess.run([self._disc_loss_optim, self._completed_images_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, global_step=global_step)
        if global_step % 100 == 0:
            _, summary = self._sess.run([self._disc_loss_optim, self._disc_loss_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, global_step=global_step)
        else:
            self._sess.run(self._disc_loss_optim, feed_dict=feed_dict)

    def train_completion_network_and_discriminator_jointly(self, images, mask_c, local_area_top_left_c, local_area_top_left_d):
        """

        :param images: 4-D array or tensor
        :param mask_c: 4-D array or tensor, has same shape of images
        :param local_area_top_left_c: 1*2 vector, top left point position of local area of mask_c in images
        :param local_area_top_left_d: 1*2 vector, top left point position of local area of mask_d in images
        :return:
        """
        global_step = self._global_step.eval(self._sess)
        feed_dict = {self._ground_truth_ph: images, self._mask_c_ph: mask_c,
                     self._local_area_top_left_c_ph: local_area_top_left_c,
                     self._local_area_top_left_d_ph: local_area_top_left_d}
        if global_step % 1000 == 0:
            _, summary = self._sess.run([self._disc_loss_optim_alpha, self._completed_images_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, global_step=global_step)
            self._sess.run(self._comp_loss_optim, feed_dict=feed_dict)
        if global_step % 100 == 0:
            _, summary = self._sess.run([self._disc_loss_optim_alpha, self._disc_loss_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, global_step=global_step)
            _, summary = self._sess.run([self._comp_loss_optim, self._comp_loss_summary], feed_dict=feed_dict)
            self.summary_writer.add_summary(summary, global_step=global_step)
        else:
            self._sess.run(self._disc_loss_optim_alpha, feed_dict=feed_dict)
            self._sess.run(self._comp_loss_optim, feed_dict=feed_dict)

    def _completion_network(self, masked_images):
        """
        create completion network model
        :param masked_images: 4-D array or tensor
        :return:
        """
        with tf.variable_scope("completion_network"):
            result = tf.layers.conv2d(masked_images, filters=64, kernel_size=5, strides=1, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=128, kernel_size=3, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=128, kernel_size=3, strides=1, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=256, kernel_size=3, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=256, kernel_size=3, strides=1, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=256, kernel_size=3, strides=1, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            "atrous conv2d"
            result = tf.layers.conv2d(result, filters=256, kernel_size=3, strides=1, dilation_rate=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=256, kernel_size=3, strides=1, dilation_rate=4, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=256, kernel_size=3, strides=1, dilation_rate=8, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=256, kernel_size=3, strides=1, dilation_rate=16, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=256, kernel_size=3, strides=1, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=256, kernel_size=3, strides=1, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d_transpose(result, filters=128,  kernel_size=4, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=128, kernel_size=3, strides=1, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d_transpose(result, filters=64, kernel_size=4, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=32, kernel_size=3, strides=1, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, filters=3, kernel_size=3, strides=1, padding="SAME")
            result = tf.tanh(result)
            result = (result + 1.0) / 2.0

            return result

    def _local_discriminator(self, images):
        """
        create local discriminator model
        :param images: 4-D array or tensor
        :return:
        """
        with tf.variable_scope("local_discriminator"):
            result = tf.layers.conv2d(images, 64, 5, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, 128, 5, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, 256, 5, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, 512, 5, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, 512, 5, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.flatten(result)
            result = tf.layers.dense(result, units=1024)
            return result

    def _global_discriminator(self, images):
        """
        create global discriminator model
        :param images: 4-D array or tensor
        :return:
        """
        with tf.variable_scope("global_discriminator"):
            result = tf.layers.conv2d(images, 64, 5, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, 128, 5, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, 256, 5, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, 512, 5, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, 512, 5, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.conv2d(result, 512, 5, strides=2, padding="SAME")
            result = tf.nn.leaky_relu(tf.layers.batch_normalization(result))

            result = tf.layers.flatten(result)
            result = tf.layers.dense(result, units=1024)
            return result

    def _discriminator(self, images, images_local_area, reuse=False):
        """
        concat the output of local discriminator and global discriminator
        :param images: 4-D array or tensor
        :param images_local_area: 4-D array or tensor, input of local discriminator
        :param reuse: bool, reuse model for discriminate ground true and inpainted images
        :return:
        """
        with tf.variable_scope("discriminator", reuse=reuse):
            local_d = self._local_discriminator(images_local_area)
            global_d = self._global_discriminator(images)
            result = tf.concat([local_d, global_d], axis=1)
            result = tf.layers.dense(result, units=1)
            return result

    def _mse_loss(self):
        """
        get mse loss for training completion network
        :return:
        """
        mse_loss = tf.multiply(1-self._mask_c_ph, self._ground_truth_ph-self._completed_images)
        mse_loss = tf.reduce_mean(tf.square(mse_loss), axis=[0, 1, 2, 3])
        return mse_loss

    def _discriminator_loss(self, ground_truth_logits, completed_image_logits):
        """
        get discriminator loss for training local discriminator and global discriminator jointly
        :param ground_truth_logits: 4-D array or tensor
        :param completed_image_logits: 4-D array or tensor, has same shape of ground_truth
        :return:
        """
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=ground_truth_logits,
                                                                             labels=tf.ones_like(ground_truth_logits)))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=completed_image_logits,
                                                                             labels=tf.zeros_like(completed_image_logits)))
        d_loss = d_loss_real + d_loss_fake
        return d_loss

    def _completion_network_loss(self, completed_images_logits):
        """
        get loss for training completion network, local discriminator and global discriminator jointly
        :param completed_images_logits: column vector, output of discriminator of completed images
        :return:
        """
        c_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=completed_images_logits,
                                                                             labels=tf.ones_like(completed_images_logits)))
        mse = self._mse_loss()

        c_loss = mse + self._alpha * c_loss_fake
        return c_loss

    def _get_local_area(self, images, local_area_top_left):
        """
        get the local area from images as input of local discriminator
        :param images: 4-D array or tensor
        :param local_area_top_left: 4-D array or tensor
        :return:
        """
        local_area = images[:, local_area_top_left[0]:local_area_top_left[0]+self._local_area_shape[0],
                     local_area_top_left[1]:local_area_top_left[1]+self._local_area_shape[1], :]
        shape = [local_area.shape[0], self._local_area_shape[0], self._local_area_shape[1], local_area.shape[-1]]
        local_area = tf.reshape(local_area, shape)
        return local_area

    def _build_graph_for_training(self):
        """
        build graph for training
        :return:
        """
        masked_images = self.mask_image(self._ground_truth_ph, self._mask_c_ph)
        completion_network_output = self._completion_network(masked_images)
        # restore image data out of completion region
        self._completed_images = self._combine_masked_image_and_com_output(masked_images, completion_network_output, self._mask_c_ph)
        completed_images_local_area = self._get_local_area(self._completed_images, self._local_area_top_left_c_ph)
        ground_truth_local_area = self._get_local_area(self._ground_truth_ph, self._local_area_top_left_d_ph)
        ground_truth_logits = self._discriminator(self._ground_truth_ph, ground_truth_local_area)
        completed_image_logits = self._discriminator(self._completed_images, completed_images_local_area, reuse=True)
        "--------loss---------"
        mse_loss = self._mse_loss()
        disc_loss = self._discriminator_loss(ground_truth_logits, completed_image_logits)
        disc_loss_alpha = self._alpha * self._discriminator_loss(ground_truth_logits, completed_image_logits)
        comp_loss = self._completion_network_loss(completed_image_logits)
        "-----optimizer-------"
        disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        comp_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='completion_network')

        self._global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

        self._mse_loss_optim = tf.train.AdamOptimizer(self._learning_rate_comp, beta1=0.5).minimize(mse_loss, var_list=comp_vars, global_step=self._global_step)
        self._disc_loss_optim = tf.train.AdamOptimizer(self._learning_rate_disc, beta1=0.5).minimize(disc_loss, var_list=disc_vars, global_step=self._global_step)
        self._disc_loss_optim_alpha = tf.train.AdamOptimizer(self._learning_rate_disc, beta1=0.5).minimize(disc_loss_alpha, var_list=disc_vars, global_step=self._global_step)
        self._comp_loss_optim = tf.train.AdamOptimizer(self._learning_rate_comp, beta1=0.5).minimize(comp_loss, var_list=comp_vars)
        "----------summary--------"
        self._mse_loss_summary = tf.summary.scalar("mse loss", mse_loss)
        self._disc_loss_summary = tf.summary.scalar("discriminator loss", disc_loss)
        self._comp_loss_summary = tf.summary.scalar("completion network loss", comp_loss)
        self._completed_images_summary = tf.summary.image("completed images", self._completed_images)

    def _build_graph_for_predicting(self):
        """
        build graph for predicting
        :return:
        """
        masked_images = self.mask_image(self._ground_truth_ph, self._mask_c_ph)
        completion_network_output = self._completion_network(masked_images)
        # restore image data out of completion region
        self._completed_images = self._combine_masked_image_and_com_output(masked_images, completion_network_output,
                                                                          self._mask_c_ph)

    def _combine_masked_image_and_com_output(self, masked_image, completion_network_output, mask):
        """
        combine completion output and  masked image to get inpainting result
        :param masked_image: 4-D array or tensor
        :param completion_network_output: 4-D array or tensor, has same shape of masked_image
        :param mask: 4-D array or tensor, has same shape of masked_image
        :return:
        """
        "restore image data out of completion region"
        completed_images = masked_image*mask + completion_network_output*(1-mask)
        return completed_images

    def mask_image(self, image, mask):
        """
        mask images
        :param image: 4-D array or tensor
        :param mask: 4-D array or tensor, has same shape of images
        :return:
        """
        masked_image = image * mask + (1 - mask) * self._gray_image_value
        return masked_image

    def complete_image(self, masked_images, mask):
        """
        complete image by using trained model
        :param masked_images: 4-D array or tensor
        :param mask: 4-D array or tensor, has same shape of images
        :return:
        """
        feed_dict = {self._ground_truth_ph: masked_images, self._mask_c_ph: mask}
        result = self._sess.run(self._completed_images, feed_dict=feed_dict)
        return result

    def save_model(self, epoch):
        """
        save trained model
        :param epoch: a int, epoch, as global step
        :return:
        """
        saver = tf.train.Saver()
        if not os.path.exists(self._ckpt_dir):
            os.mkdir(self._ckpt_dir)
        ckpt_path = os.path.join(self._ckpt_dir, self._model_name)
        saver.save(self._sess, ckpt_path, global_step=epoch)
        print('save ckpt:%s\n' % ckpt_path)