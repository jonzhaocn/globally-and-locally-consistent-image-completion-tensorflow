# globally-and-locally-consistent-image-completion-tensorflow
A tensorflow implement of [Globally and locally consistent image completion](https://dl.acm.org/citation.cfm?id=3073659).

## Version

* `python`		3.5.4
* `tensorflow`  1.4.0

## Examples

#### Celeba Data

* Use CelebA dataset to train the network.
* Image preprocessing: crop the central part(128×128) of image and mask the central part (64×64) of cropped image.

* The completed images are the output of completion network that trained 20 epoch.

<p align="center">
	<a href="/image/celeba/1_masked.png" target="_blank"><img src="/image/celeba/1_masked.png" height="128px" width="128px"></a>
	<a href="/image/celeba/1_completed.png" target="_blank"><img src="/image/celeba/1_completed.png" height="128px" width="128px"></a>
    <a href="/image/celeba/2_masked.png" target="_blank"><img src="/image/celeba/2_masked.png" height="128px" width="128px"></a>
	<a href="/image/celeba/2_completed.png" target="_blank"><img src="/image/celeba/2_completed.png" height="128px" width="128px"></a>
</p>

<p align="center">
	<a href="/image/celeba/3_masked.png" target="_blank"><img src="/image/celeba/3_masked.png" height="128px" width="128px"></a>
	<a href="/image/celeba/3_completed.png" target="_blank"><img src="/image/celeba/3_completed.png" height="128px" width="128px"></a>
    <a href="/image/celeba/4_masked.png" target="_blank"><img src="/image/celeba/4_masked.png" height="128px" width="128px"></a>
	<a href="/image/celeba/4_completed.png" target="_blank"><img src="/image/celeba/4_completed.png" height="128px" width="128px"></a>
</p>

<p align="center">
	<a href="/image/celeba/5_masked.png" target="_blank"><img src="/image/celeba/5_masked.png" height="128px" width="128px"></a>
	<a href="/image/celeba/5_completed.png" target="_blank"><img src="/image/celeba/5_completed.png" height="128px" width="128px"></a>
    <a href="/image/celeba/6_masked.png" target="_blank"><img src="/image/celeba/6_masked.png" height="128px" width="128px"></a>
	<a href="/image/celeba/6_completed.png" target="_blank"><img src="/image/celeba/6_completed.png" height="128px" width="128px"></a>
</p>

<p align="center">
	<a href="/image/celeba/7_masked.png" target="_blank"><img src="/image/celeba/7_masked.png" height="128px" width="128px"></a>
	<a href="/image/celeba/7_completed.png" target="_blank"><img src="/image/celeba/7_completed.png" height="128px" width="128px"></a>
    <a href="/image/celeba/8_masked.png" target="_blank"><img src="/image/celeba/8_masked.png" height="128px" width="128px"></a>
	<a href="/image/celeba/8_completed.png" target="_blank"><img src="/image/celeba/8_completed.png" height="128px" width="128px"></a>
</p>
<p align="center">
	<a href="/image/celeba/9_masked.png" target="_blank"><img src="/image/celeba/9_masked.png" height="128px" width="128px"></a>
	<a href="/image/celeba/9_completed.png" target="_blank"><img src="/image/celeba/9_completed.png" height="128px" width="128px"></a>
    <a href="/image/celeba/10_masked.png" target="_blank"><img src="/image/celeba/10_masked.png" height="128px" width="128px"></a>
	<a href="/image/celeba/10_completed.png" target="_blank"><img src="/image/celeba/10_completed.png" height="128px" width="128px"></a>
</p>



#### ImageNet Data

- Using ImageNet dataset to train the network.
- Image preprocessing: resize images so that the smallest edge is a random value in the [128, 200] , crop a random part(128×128) of image and mask the central part (64×64) of cropped image.

- The completed images are the output of completion network that trained 50 epoch.

<p align="center">
	<a href="/image/imagenet/1_masked.png" target="_blank"><img src="/image/imagenet/1_masked.png" height="128px" width="128px"></a>
	<a href="/image/imagenet/1_completed.png" target="_blank"><img src="/image/imagenet/1_completed.png" height="128px" width="128px"></a>
    <a href="/image/imagenet/2_masked.png" target="_blank"><img src="/image/imagenet/2_masked.png" height="128px" width="128px"></a>
	<a href="/image/imagenet/2_completed.png" target="_blank"><img src="/image/imagenet/2_completed.png" height="128px" width="128px"></a>
</p>

<p align="center">
	<a href="/image/imagenet/3_masked.png" target="_blank"><img src="/image/imagenet/3_masked.png" height="128px" width="128px"></a>
	<a href="/image/imagenet/3_completed.png" target="_blank"><img src="/image/imagenet/3_completed.png" height="128px" width="128px"></a>
    <a href="/image/imagenet/4_masked.png" target="_blank"><img src="/image/imagenet/4_masked.png" height="128px" width="128px"></a>
	<a href="/image/imagenet/4_completed.png" target="_blank"><img src="/image/imagenet/4_completed.png" height="128px" width="128px"></a>
</p>

<p align="center">
	<a href="/image/imagenet/5_masked.png" target="_blank"><img src="/image/imagenet/5_masked.png" height="128px" width="128px"></a>
	<a href="/image/imagenet/5_completed.png" target="_blank"><img src="/image/imagenet/5_completed.png" height="128px" width="128px"></a>
    <a href="/image/imagenet/6_masked.png" target="_blank"><img src="/image/imagenet/6_masked.png" height="128px" width="128px"></a>
	<a href="/image/imagenet/6_completed.png" target="_blank"><img src="/image/imagenet/6_completed.png" height="128px" width="128px"></a>
</p>

<p align="center">
	<a href="/image/imagenet/7_masked.png" target="_blank"><img src="/image/imagenet/7_masked.png" height="128px" width="128px"></a>
	<a href="/image/imagenet/7_completed.png" target="_blank"><img src="/image/imagenet/7_completed.png" height="128px" width="128px"></a>
    <a href="/image/imagenet/8_masked.png" target="_blank"><img src="/image/imagenet/8_masked.png" height="128px" width="128px"></a>
	<a href="/image/imagenet/8_completed.png" target="_blank"><img src="/image/imagenet/8_completed.png" height="128px" width="128px"></a>
</p>

<p align="center">
	<a href="/image/imagenet/9_masked.png" target="_blank"><img src="/image/imagenet/9_masked.png" height="128px" width="128px"></a>
	<a href="/image/imagenet/9_completed.png" target="_blank"><img src="/image/imagenet/9_completed.png" height="128px" width="128px"></a>
    <a href="/image/imagenet/10_masked.png" target="_blank"><img src="/image/imagenet/10_masked.png" height="128px" width="128px"></a>
	<a href="/image/imagenet/10_completed.png" target="_blank"><img src="/image/imagenet/10_completed.png" height="128px" width="128px"></a>
</p>

## References

[1].  `Iizuka Satoshi, Simo-Serra Edgar, and Ishikawa Hiroshi. Globally and locally consistent image completion[M]. ACM, 2017, 36(4):1-14`

[2].  `Goodfellow I J, Pougetabadie J, Mirza M, et al. Generative Adversarial Networks[J]. Advances in Neural Information Processing Systems, 2014, 3:2672-2680.`

[3].  `Liu Z, Luo P, Wang X, et al. Deep learning face attributes in the wild[C]//Proceedings of the IEEE International Conference on Computer Vision. 2015: 3730-3738.`

[4]. `Deng J, Dong W, Socher R, et al. Imagenet: A large-scale hierarchical  image database[C]//Computer Vision and Pattern Recognition, 2009. CVPR 2009. IEEE Conference on. Ieee, 2009: 248-255.`
