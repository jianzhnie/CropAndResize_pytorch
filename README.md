# CropAndResize for PyTorch
 This implementation is PyTorch version of `crop_and_resize` and supports both forward and backward on CPU and GPU. It's faster than PIL based crop and resize.

## Introduction
The `crop_and_resize` function is ported from [tensorflow](https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize),
and has the same interface with tensorflow version, except the input feature map
should be in `NCHW` order in PyTorch.
They also have the same output value (error < 1e-5) for both forward and backward as we expected.

**Note:**
Document of `crop_and_resize` can be found [here](https://www.tensorflow.org/api_docs/python/tf/image/crop_and_resize).Notice that `crop_and_resize` use *normalized `(y1, x1, y2, x2)`* as input).

**Warning:**
Currently it only works using the default GPU (index 0)

## Usage

+ Install the package and run test.

    ```sh
    cd CropAndResize.pytorch
    ## make install
    python setup.py install
    ## run test
    sh test.sh
    ```

+ Use crop_and_resize function for student behavior classification.

    ```sh
    cd body_action_classifier
    python demo.py
    ```
+ When batch_size > 1, the processing is slightly different, please check it in minibatch_demo.py.
    ```sh
    cd body_action_classifier
    python minibatch_demo.py
    ```

## Wiki

The result of `crop_and_resize` may different from original PIL based function, you can find the experimental result and there difference [here](https://wiki.dm-ai.cn/pages/viewpage.action?pageId=71787887#).
