"""
ESP32 Keyword Spotting – End‑to‑End Training Script (MicroKWS-based)
-------------------------------------------------------------------
This module provides a reproducible, notebook‑free pipeline to train, validate,
and export a compact Keyword Spotting (KWS) model suitable for embedded targets
(ESP32 / microTVM / TFLite Micro). It mirrors the logic of the original Jupyter
notebook while making it easier to run from the command line and to integrate
with CI. The code assumes MFCC/log‑mel style features (a.k.a. the “micro”
preprocess), lightweight CNN/DS‑CNN‑style architectures, and standard practices
for dataset preparation, augmentation, and checkpointing.

Key capabilities
- Dataset ingestion & preprocessing via the project’s `data.py` utilities
  (silence/unknown handling, background noise mixing, time shift).
- Model construction via `models.py` (wires in your student architecture under
  `student/model.py`) with shapes derived from `prepare_model_settings`.
- Training loop with piecewise LR schedule, validation, early stopping /
  checkpointing hooks, and test‑set evaluation.
- Optional conversion and sanity checks downstream via the deploy scripts
  (e.g., `convert.py` and `test_model_tflite.py`).

How to use
1) Activate your virtual environment and install dependencies:
   $ source ~/.venvs/esp32-kws/bin/activate
   $ python -m pip install -r requirements.txt

2) Run training from the project root (example settings shown):
   $ python MicroKWS.py \
       --data_dir /path/to/speech_dataset \
       --wanted_words yes,no,up,down,left,right,on,off,stop,go \
       --sample_rate 16000 --clip_duration_ms 1000 \
       --window_size_ms 30 --window_stride_ms 20 \
       --dct_coefficient_count 40 \
       --how_many_training_steps 15000,3000 \
       --learning_rate 0.001,0.0001 \
       --batch_size 100 --model_architecture micro_kws_student

3) After training, you can convert to TFLite using `convert.py`, and test with
   `test_model_tflite.py` or `label_wav.py` for single‑file inference checks.

Notes
- Keep virtual environments on the Linux filesystem when using WSL for reliable
  builds (avoid creating venvs on /mnt/c).
- For embedded deployment, ensure your label order, feature parameters
  (n_mels/MFCC, window size/stride, sample rate), and tensor shapes are
  exported alongside the model to keep training ↔ firmware consistent.
"""

#!/usr/bin/env python
# coding: utf-8

# # MicroKWS Training Flow

# This notebook should give an introduction on the required procedure to design, train and quantize a small machine learning model using the Tensorflow Lite and the Keras API. The application example is a keyword-spotting (KWS) task which should ideally be suitable to run on an energy efficent device e.g. a small microcontroller platform (see next lab).
# 
# In the following a step by step guide is provided. Please follow the notebooks contents sequentially by executiong one cell after each other while inspecting the used python code as well as the program outputs printed to the screen. From time to time theoretical tasks are going to be introduced. Please try to answer them **without changing the contents of the previous cells**. At the end of this document, there is a Programming Challenge which has to be solved alongside with the previous theoretical questions to pass this lab assignment. As this challenge involves some programming, it is recommended to duplicate this notebook before starting to play around with the code in the allowed cells.
# 
# If you have never heard of *Jupyter Notebooks* before, please first have a look at the Setup section in the Lab Manual and check out https://docs.jupyter.org/en/latest/start/index.html for more information.

# ### Disclaimer
# 
# This tutorial is inpired by the contents of: https://github.com/ARM-software/ML-examples/tree/main/tflu-kws-cortex-m/Training

# ## 0. Install software

# The following steps should ideally done before launching this Jupyter notebook! (See `README.md`!)

# **1. Clone repository**
# 
# ```
# git clone git@gitlab.lrz.de:de-tum-ei-eda-esl/ESD4ML/micro-kws.git
# ```
# 
# 
# **2. Create virtual python environment**
# 
# ```
# virtualenv -p python3.8 venv
# ```
# 
# **3. Enter virtual python environment**
# 
# ```
# source venv/bin/activate
# ```
# 
# **4. Enter directory**
# 
# ```
# cd micro-kws/1_train
# ```
# 
# **5. Install python packages into environment**
# 
# ```
# pip install -r requirements.txt
# ```
# 
# **6. Start jupyter notebook**
#     
# ```
# jupyter notebook Flow.ipynb
# ```
# 
#   If using a remote host, append: ` --no-browser --ip 0.0.0.0 --port XXXX` (where XXXX should be a number greater than 1000)
#   
#   If you experience warnings it might help to use ` --NotebookApp.iopub_msg_rate_limit=1.0e10  --NotebookApp.iopub_data_rate_limit=1.0e10`

# The following "IPython magic" allows editing python files without restarting the Jupyter kernel.

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ## 1. Python imports

# Python builtin dependencies

# In[2]:


import os
import tempfile

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"  # Reduce verbosity
import argparse
from pathlib import Path


# Third party dependencies

# In[3]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# Jupyter specific

# In[3]:


from IPython.display import FileLink


# Helper scripts

# In[ ]:


import data
import models
from test_model import get_accuracy, get_confusion_matrix
from test_model_tflite import tflite_test
from estimate import load_model, estimate_model_macs, estimate_model_rom, estimate_model_ram


# Import student code

# In[ ]:


from student.metrics import get_student_metrics
from student.callbacks import get_student_callbacks


# ## 2. Define training parameters

# In this section the hyperparameters for the model and training procedure are defined. Feel free to read through the code line by line as the options should be all documented well. You are NOT supposed to change any parameters except the following:
# 
# - `FLAGS.model_name`: This will be used as the base filename when writing the converted model to the disk. Change this from `"micro_kws_xs"` to `micro_kws_student` when starting the Programming challenge.
# - `FLAGS.data_dir`: May be changed to a persitent directory to keep the downloaded dataset between reboots if working on a personal machine. If working on a chair computer, feel free to change this to `/usr/local/labs/ESD4ML/current/common/data/speech_commands_v0.02` to skip the download procedure.
# - `FLAGS.wanted_words`: should be changed to the set of keywords which was assigned to your group **after** answering the theoretical questions on the default pair of keywords.

# In[ ]:


FLAGS = argparse.Namespace()

# Overwrite the model name provided by keras with a custom one
FLAGS.model_name = "micro_kws_xs"

# Location of speech training data archive on the web.
FLAGS.data_url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

# Where to download the speech training data to.
try:
    login = os.getlogin()
except:
    login = "unknown"
FLAGS.data_dir = os.getenv(
    "SPEECH_COMMANDS_DIR",
    default=os.path.join(tempfile.gettempdir(), login, "speech_dataset"),
)

# Words to use (others will be added to an unknown label)
FLAGS.wanted_words = "yes,no"

# Total number of classifiations labels (wanted_words + silence + unknown)
FLAGS.num_classes = len(FLAGS.wanted_words.split(",")) + 2

# How loud the background noise should be, between 0 and 1.
FLAGS.background_volume = 0.1

# How many of the training samples have background noise mixed in.
FLAGS.background_frequency = 0.8

# How much of the training data should be silence.
FLAGS.silence_percentage = 100.0 / FLAGS.num_classes

# How much of the training data should be unknown words
FLAGS.unknown_percentage = 100.0 / FLAGS.num_classes

# Range to randomly shift the training audio by in time.
FLAGS.time_shift_ms = 100.0

# What percentage of wavs to use as a test set.
FLAGS.testing_percentage = 10

# What percentage of wavs to use as a validation set.
FLAGS.validation_percentage = 10

# Expected sample rate of the wavs
FLAGS.sample_rate = 16000

# Expected duration in milliseconds of the wavs
FLAGS.clip_duration_ms = 1000

# How long each spectrogram timeslice is
FLAGS.window_size_ms = 30.0

# How long each spectrogram timeslice is
FLAGS.window_stride_ms = 20.0

# How many bins to use for the MFCC fingerprint
FLAGS.dct_coefficient_count = 40

# How many training loops to run
FLAGS.how_many_training_steps = "12000,3000"

# How often to evaluate the training results.
FLAGS.eval_step_interval = 400

# How large a learning rate to use when training.
FLAGS.learning_rate = "0.001,0.0001"

# How many items to train with at once
FLAGS.batch_size = 100

# Where to save summary logs for TensorBoard.
# FLAGS.summaries_dir = '/tmp/retrain_logs'

# Directory to write event logs and checkpoint.
FLAGS.train_dir = "training"

# Directory to write converted models to.
FLAGS.models_dir = "models"


# ## 3. Create Keras Model

# Get the model settings as they are required for the preprocessing, training and quantization.

# In[ ]:


model_settings = models.prepare_model_settings(
    len(data.prepare_words_list(FLAGS.wanted_words.split(","))),
    FLAGS.sample_rate,
    FLAGS.clip_duration_ms,
    FLAGS.window_size_ms,
    FLAGS.window_stride_ms,
    FLAGS.dct_coefficient_count,
)


# Define a model architecture using the Keras API. A predefined model can be found in `models.py`. The model for the final challenge has to be defined in `student/model.py`

# The following is just an example on how to define a minimal model architecture for the MicroTVM application:
# 
# ```python
# def create_micro_kws_xs_model(model_settings):
#     """Builds a model with a single depthwise-convolution layer followed by a single fully-connected layer.
#     Args:
#         model_settings: Dict of different settings for model training.
#     Returns:
#         tf.keras Model of the 'micro_kws_xs' architecture.
#     """
# 
#     # Get relevant model setting.
#     input_frequency_size = model_settings["dct_coefficient_count"]
#     input_time_size = model_settings["spectrogram_length"]
# 
#     inputs = tf.keras.Input(shape=(model_settings["fingerprint_size"]), name="input")
# 
#     # Reshape the flattened input.
#     x = tf.reshape(inputs, shape=(-1, input_time_size, input_frequency_size, 1))
# 
#     # First convolution.
#     x = tf.keras.layers.DepthwiseConv2D(
#         depth_multiplier=4,
#         kernel_size=(5, 4),
#         strides=(2, 2),
#         padding="SAME",
#         activation="relu",
#     )(x)
# 
#     # Flatten for fully connected layers.
#     x = tf.keras.layers.Flatten()(x)
# 
#     # Output fully connected.
#     output = tf.keras.layers.Dense(units=model_settings["label_count"], activation="softmax")(x)
# 
#     return tf.keras.Model(inputs, output, name=FLAGS.model_name)
# ```

# Generate keras model. The `model.summary()` utility provides a way to inspect the layers of Keras model with its shapes and parameters.

# In[ ]:


model = models.get_model(model_settings, FLAGS.model_name, model_name=FLAGS.model_name)
model.summary()


# ## 4. Prepare dataset

# While keyword-spotting is quite simple tasks, the preprocessing to generate input features for training is non-trivial. Hence, the implementation of the `Audioprocessor()` class is omited here. If interested, check out the [`data.py`](./data.py) script for more information.
# 
# One aspect, which is very important here is the `micro=True` option as it ensures that the same preprocessing (conversion of input WAV files to an Image) is applied to the input dataset as used in the mcirocontroller target software.

# In[ ]:


audio_processor = data.AudioProcessor(
    data_url=FLAGS.data_url,
    data_dir=FLAGS.data_dir,
    silence_percentage=FLAGS.silence_percentage,
    unknown_percentage=FLAGS.unknown_percentage,
    wanted_words=FLAGS.wanted_words.split(","),
    validation_percentage=FLAGS.validation_percentage,
    testing_percentage=FLAGS.testing_percentage,
    model_settings=model_settings,
    micro=True,
)


# Let's define a helper function to visualize some features:

# In[ ]:


def visualize_feature(feature):
    # Utility to display a given feature from the dataset inside the notebook
    feature_data, feature_label = feature

    feature_data = feature_data.numpy()
    feature_label = feature_label.numpy()

    feature_label_str = (["silence", "unknown"] + FLAGS.wanted_words.split(","))[feature_label]

    feature_reshaped = np.reshape(feature_data, (49, 40)).T

    p = plt.imshow(feature_reshaped, cmap="gray", vmin=0, vmax=26)
    plt.title(f"Label: {feature_label_str}")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [ƒ]")


# Execute the following cell a few times to inpect the generated features for some keywords/labels.

# In[ ]:


feature = (
    audio_processor.get_data(audio_processor.Modes.VALIDATION)
    .shuffle(100)
    .take(1)
    .get_single_element()
)
visualize_feature(feature)


# ## 5. Run Training

# Define training procedure using the previously defined parameters.
# 
# In addition to the training hyperparamerters (training steps, learning rate,...) an optimizer (`Adam`), a loss function (`SparseCategoricalCrossentropy`) as well as a metric is selected for the training and passed to the `model.compile()` method.
# 
# The actual training happens when `model.fit()` is called. The training progress should be visible on the screen. While multiple epochs ("one pass over the entire dataset") are required for the training, the validation accuracy is evaluated every 200 steps and the weights are written to a directory automatically.
# 
# At the end of the training procedure the final test accuracy of the trained model is printed to the screen.

# In[ ]:


def train(model, audio_processor):
    # We decay learning rate in a constant piecewise way to help learning.
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(",")))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(",")))
    lr_boundary_list = training_steps_list[:-1]  # Only need the values at which to change lr.
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_boundary_list, values=learning_rates_list
    )

    # Specify the optimizer configurations.
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Compile the model.
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Prepare/split the dataset.
    train_data = audio_processor.get_data(
        audio_processor.Modes.TRAINING,
        FLAGS.background_frequency,
        FLAGS.background_volume,
        int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000),
    )
    train_data = train_data.repeat().batch(FLAGS.batch_size).prefetch(tf.data.AUTOTUNE)
    val_data = audio_processor.get_data(audio_processor.Modes.VALIDATION)
    val_data = val_data.batch(FLAGS.batch_size).prefetch(tf.data.AUTOTUNE)

    # We train for a max number of iterations so need to calculate how many 'epochs' this will be.
    training_steps_max = np.sum(training_steps_list)
    training_epoch_max = int(np.ceil(training_steps_max / FLAGS.eval_step_interval))

    # Callbacks.
    train_dir = Path(FLAGS.train_dir) / FLAGS.model_name / "best"
    train_dir.mkdir(parents=True, exist_ok=True)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=(train_dir / (FLAGS.model_name + "_{val_accuracy:.3f}_ckpt")),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )

    # Train the model.
    model.fit(
        x=train_data,
        steps_per_epoch=FLAGS.eval_step_interval,
        epochs=training_epoch_max,
        validation_data=val_data,
        callbacks=[model_checkpoint_callback, *get_student_callbacks()],
    )

    # Test and save the model.
    test_data = audio_processor.get_data(audio_processor.Modes.TESTING)
    test_data = test_data.batch(FLAGS.batch_size)

    # Evaluate the model performace.
    test_loss, test_acc = model.evaluate(x=test_data)
    print(f"Final test accuracy: {test_acc*100:.2f}%")


# Invoke training procedure (**Warning:** This will take a very long time!)

# In[ ]:


train(model, audio_processor)


# Determine latest checkpoint

# In[ ]:


latest = tf.train.latest_checkpoint(Path(FLAGS.train_dir) / FLAGS.model_name / "best")
print(latest)


# Pick a checkpoint

# In[ ]:


FLAGS.checkpoint = latest  # Feel free to choose a different one!


# **Task:** Update [`student/callbacks.py`](./student/callbacks.py) to define an EarlyStopping (https://keras.io/api/callbacks/early_stopping/) callback with keras which stops the training procedure after 10 or more epochs of neglectible improvement of the `val_loss` quantity. Then rerun the training procedure.

# ## 6. Test trained TensorFlow model

# Define test procedure we can use to evaluate our models performance.
# 
# The used test routines are defined in [`test_model.py`](./test_model.py).

# In[ ]:


def test(model, audio_processor, model_settings):
    """Calculate accuracy and confusion matrices on validation and test sets.

    Model is created and weights loaded from supplied command line arguments.
    """
    model.load_weights(FLAGS.checkpoint).expect_partial()

    # Get test data
    data = audio_processor.get_data(audio_processor.Modes.TESTING).batch(FLAGS.batch_size)

    # Invoke model
    predictions = model.predict(data)

    # Calculate indices
    expected_indices = np.concatenate([y for x, y in data])
    predicted_indices = tf.argmax(predictions, axis=1)

    print("Running testing on test set...")
    accuracy = get_accuracy(expected_indices, predicted_indices)
    confusion_matrix = get_confusion_matrix(expected_indices, predicted_indices, model_settings)

    # Print accuracy and confusion matrix
    print(
        f"test accuracy = {accuracy * 100:.2f}%"
        f"(N={audio_processor.set_size(audio_processor.Modes.TESTING)})"
    )
    print()
    print("confusion matrix:")
    print(confusion_matrix.numpy())

    # Print student metrics
    print()
    print("metrics:")
    words = ["silence", "unknown"] + FLAGS.wanted_words.split(",")
    for idx, label in enumerate(words):
        data = get_student_metrics(confusion_matrix, idx)

        # Filter None values
        data = {key: value for key, value in data.items() if value is not None}
        if len(data) == 0:
            continue
        print(f"  {label}:")
        for key, value in data.items():
            if isinstance(value, float):
                value = f"{value:.3f}"
            print(f"    {key} = {value}")
        print()


# Run test procedure

# In[ ]:


test(model, audio_processor, model_settings)


# Confusion matrices are also printed as they provide infomation about how the individual classes have performed.

# **Task:** After completing the programming tasks in `student/metrics.py` the per-class recall, precision and f1-score will be printed above as well.

# ## 7. Quantization and Conversion to TFLite 

# Define conversion procedure using the `TFLiteConverter` which creates a `.tflite` file which holds the model graph and constant weights.

# In[ ]:


NUM_REP_DATA_SAMPLES = (
    100  # Number of representative samples which will be used for the post-training quantization.
)


def convert(model, audio_processor, checkpoint, quantize, inference_type, tflite_path):
    """Load our trained floating point model and convert it.
    TFLite conversion or post training quantization is performed and the
    resulting model is saved as a TFLite file.
    We use samples from the validation set to do post training quantization.
    Args:
        model: The keras model.
        audio_processor: Audio processor class object.
        checkpoint: Path to training checkpoint to load.
        quantize: Whether to quantize the model or convert to fp32 TFLite model.
        inference_type: Input/output type of the quantized model.
        tflite_path: Output TFLite file save path.
    """
    model.load_weights(checkpoint).expect_partial()

    val_data = audio_processor.get_data(audio_processor.Modes.VALIDATION).batch(1)

    def _rep_dataset():
        """Generator function to produce representative dataset."""
        i = 0
        for mfcc, label in val_data:
            if i > NUM_REP_DATA_SAMPLES:
                break
            i += 1
            yield [mfcc]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if quantize:
        # Quantize model and save to disk.
        if inference_type == "int8":
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        # Int8 post training quantization needs representative dataset.
        converter.representative_dataset = _rep_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print("{} model saved to {}.".format("Quantized" if quantize else "Converted", tflite_path))


# Invoke the previously defined conversion routine

# In[ ]:


Path(FLAGS.models_dir).mkdir(exist_ok=True)
keywords_str = FLAGS.wanted_words.replace(",", "")
tflite_path_quantized = (
    Path(FLAGS.models_dir) / f"{FLAGS.model_name}_{keywords_str}_quantized.tflite"
)
tflite_path = Path(FLAGS.models_dir) / f"{FLAGS.model_name}_{keywords_str}.tflite"

# Load floating point model from checkpoint and convert it.
convert(model, audio_processor, FLAGS.checkpoint, False, "fp32", tflite_path)

# Quantize model from checkpoint and convert it.
convert(model, audio_processor, FLAGS.checkpoint, True, "int8", tflite_path_quantized)


# ## 8. Test Converted TFLite Model

# Test the newly converted model on the test set.
# 
# The `tflite_test` function is defined in [`test_model_tflite.py`](./test_model_tflite.py).

# **Floating Point**

# In[ ]:


_ = tflite_test(model_settings, audio_processor, str(tflite_path), mode="test")


# **Quantized**

# In[ ]:


_ = tflite_test(model_settings, audio_processor, str(tflite_path_quantized), mode="test")


# Interestingly the accuracy of the quantized model can be even better than the floating point one. This behavior is a consequence of the quantization procedure the TFLite which requires a `representative_dataset` to partially re-train some weights during the quantization.

# ## 9. Visualize TFLite Model

# **Figure 1:** Example TFLite KWS Model
# <img src="resources/micro_kws_xs_yesno_quantized.png" alt="Netron graph" width="150"/>

# Use the following links to download the generated `.tflite` files

# **Floating Point**

# In[ ]:


FileLink(tflite_path)


# **Quantized**

# In[ ]:


FileLink(tflite_path_quantized)


# Use the web application https://netron.app/ to generate a graph representation of the converted model.

# ## 10. Performance and Memory Estimations

# In this section you will learn how to estimate the complexity of a given model architecture to evaluate if it is suitable to be deployed on a very constrained embedded device. We will also make use of some simplifications to make our life easier.

# In[ ]:


m = load_model(tflite_path_quantized)


# ### 10.1 ROM Usage

# Estimating the ROM usage of model graph is non-trivial if considering all types of data which contributes to the ROM footprint e.g.
# 
# - Code size of the implementation of neural network kernels
# - ROM overhead of the used inference engine/runtime
# - The constant weights used by the kernels/operators
# - Driver code for interfacing with peripherals such as sensors or a serial communication port
# - If not running bare-metal: Additional ROM usage depending on the used operating system or RTOS
# 
# In addition the final ROM size depends on further properties such as the used compiler flags (optimization level) and certain memory alignment requirements.

# Lets consider the previously trained `micro_kws_xs_yesno_quantized.tflite` model: It's file size is $10704$ bytes. However the TFLite Flatbuffer format also stores a compat representation of the models tensors (name, shape, type,...) and operators (inputs, outputs, parameters) and some metadata alongside with the model weights which contribute to the biggest part of the file.

# In the following we will only consider the actual weights used by the model, hence we can ignore any implementation specific overheads.
# 
# The used tensor datatypes (specifically `float32`, `int8` or `int32`) per operator have to be investigated to calculate the total amount of RAM required by the model weights. The https://netron.app/ can be used for this task.

# **Task:** Derive a formula to estimate the memory requirement to store all constant weights of the quantized model in ROM considering the used data types.

# **Task:** Update the `estimate_rom` utility in [`student/estimate.py`](./student/estimate.py) to use your derived formula.

# Execute the following cell to test run your implementation:

# In[ ]:


estimated_rom = estimate_model_rom(m)
print(f"Estimated ROM Usage: {estimated_rom/1e3:.3f} KiB")


# ### 10.2 RAM Usage

# Investigating the RAM usage of a given model involves similar challenges as the ROM-estimations.
# 
# RAM requirements can vary a lot with the chosen model architecture and deployment flow. The largest contribution to the RAM footprint are often intermediate tensor buffers (activations) but also temporary scratchpad memory required by certain kernel implementations.
# 
# Optionally memory planning can be used to reduce the RAM usage by analysing the lifetime of certain input and output buffers. This process can happen during runtime (online) or statically (offline) depending on deployment approach.
# 
# Again additional application-specifc overheads might also be non-negligible.

# **Task:** Derive formulars for estimating the dynamic memory requirement of the quantized model based on the TFLite graph only considering intermediate tensor buffers stored in RAM for optimal memory-planning.
# 
# *Assumptions:*
# - Neither branches nor nodes with multiple inputs/outputs extist in the trained model.
# - Assume that the graph is processed in a linear way so that at most 2 buffers will be used at the same time.

# **Task:** Update the `estimate_ram` utility in [`student/estimate.py`](./student/estimate.pyestimate_rom) to use your derived formulas.

# Execute the following cell to test run your implementation:

# In[ ]:


estimated_ram = estimate_model_ram(m)
print(f"Estimated RAM Usage: {estimated_ram/1e3:.3f} KiB")


# ### 10.3 Number of MAC Operations

# In this section the compute demand of a given TFLite model should be estimated.
# 
# As a first simplification we will only consider the operation which will have the biggest impact on the actual inference time: Multiply-Add (MAC)
# 
# These operations can be found in Dense (FullyConnected), and convolutional layers. Thus other operations (here: Reshape, Flatten as well as activation functions) can be neglected for the following task.
# 
# First, a formular to describe the number of MAC operations of the three major types of with repect to the given tensor dimensions and parameters.
# 
# **Example (Dense/FullyConnected):**
# 
#   Assume: $h_{out}=h_{in}$, $w_{out}=w_{filter}$
# 
#   $$num_{mac} = h_{out} \cdot w_{out} \cdot h_{filter}$$
#   
#   For the example keras model: $1 \cdot 4 \cdot 2000 \approx 8k \mathrm{MACs}$

# **Task:** Estimate the number of Multiply-Add operations used in the quantized model (see Figure 1) by deriving a formula for `num_mac` in a (depthwise) convolutional layer with respect to $$h_{kernel}, w_{kernel}, c_{in}, c_{out}, h_{out}, w_{out}$$ and (if applicable) $$depth\_multiplier,h_{stride}, w_{stride}$$.

# **Task:** Update the `estimate_fully_connected`, `estimate_conv2d_macs` and `estimate_depthwise_conv2d_macs` in [`student/estimate.py`](./student/estimate.py) to use your derived formulas.

# Execute the following cell to test run your implementation:

# In[ ]:


estimated_macs = estimate_model_macs(m)
print(f"Estimated MACs: {estimated_macs}")


# ## 11. Final challenge

# **Task:** To get bonus credits in the lab you have to design a model architecture for the keyword-spotting task which satisfies each of the following constraints:
# 
# See `Lab 1 Manual`!

# ## 12. Lab 1 Submission

# The following cell can be executed to run some basic tests on your code. The converage of these unit tests is far away from complete and 100% successful tests to not imply a correct solution.

# In[ ]:


get_ipython().system('python -m pytest tests/')


# After completing the lab execises, the following cell can be executed to generate the ZIP file containing the expected files. This script also runs some basic checks to make sure that nothing is missing in your submission. The `submission.zip` file has to be uploaded to Moodle before the deadline.

# In[ ]:


get_ipython().system('python submit.py')


# This is the end of the Notebook.

# ----------------------------
# Command-line interface (with sensible defaults)
# ----------------------------
if __name__ == "__main__":
    import os
    import argparse
    import tempfile
    from pathlib import Path
    import numpy as np
    import tensorflow as tf

    # Reduce TF log noise by default (override by unsetting this env var)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")          # hide TF info/warnings
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")           # default to CPU unless user overrides

    parser = argparse.ArgumentParser(description="Train a compact KWS model (MicroKWS-based).")
    try:
        login = os.getlogin()
    except Exception:
        login = "unknown"

    parser.add_argument("--data_url", type=str,
                        default="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz",
                        help="Location of speech training data archive on the web.")
    parser.add_argument("--data_dir", type=str,
                        default=os.getenv("SPEECH_COMMANDS_DIR",
                                          default=os.path.join(tempfile.gettempdir(), login, "speech_dataset")),
                        help="Directory for the speech commands dataset.")
    parser.add_argument("--silence_percentage", type=float, default=None,
                        help="Percentage of silence samples to include (defaults to balanced share).")
    parser.add_argument("--unknown_percentage", type=float, default=None,
                        help="Percentage of unknown samples to include (defaults to balanced share).")
    parser.add_argument("--testing_percentage", type=int, default=10,
                        help="Percentage of wavs used as test set.")
    parser.add_argument("--validation_percentage", type=int, default=10,
                        help="Percentage of wavs used as validation set.")
    parser.add_argument("--sample_rate", type=int, default=16000, help="Expected sample rate of the wavs.")
    parser.add_argument("--clip_duration_ms", type=int, default=1000, help="Clip duration in ms.")
    parser.add_argument("--window_size_ms", type=float, default=30.0, help="Spectrogram window size in ms.")
    parser.add_argument("--window_stride_ms", type=float, default=20.0, help="Spectrogram stride in ms.")
    parser.add_argument("--dct_coefficient_count", type=int, default=40, help="MFCC/log-mel feature bins.")
    parser.add_argument("--how_many_training_steps", type=str, default="15000,3000",
                        help="Comma-separated training steps for piecewise schedule.")
    parser.add_argument("--eval_step_interval", type=int, default=400,
                        help="Steps per epoch (evaluations per epoch).")
    parser.add_argument("--learning_rate", type=str, default="0.001,0.0001",
                        help="Comma-separated learning rates for piecewise schedule.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size.")
    parser.add_argument("--wanted_words", type=str,
                        default="yes,no,up,down,left,right,on,off,stop,go",
                        help="Comma-separated words to use (others become 'unknown').")
    parser.add_argument("--train_dir", type=str, default="training",
                        help="Directory to write event logs and checkpoints.")
    parser.add_argument("--model_architecture", type=str, default="micro_kws_student",
                        help="Model architecture key (see models.get_model).")
    parser.add_argument("--model_name", type=str, default="micro_kws",
                        help="Logical name for this model run (used in checkpoint path).")
    parser.add_argument("--micro", dest="micro", action="store_true", default=True,
                        help="Use micro preprocess (default True).")
    parser.add_argument("--no-micro", dest="micro", action="store_false")

    args, _ = parser.parse_known_args()

    # Lazy imports from project modules to avoid circular imports at top-level
    import models
    import data
    # Callbacks (support both names to match your repo)
    try:
        from student.callbacks import get_callbacks as _get_callbacks
    except Exception:
        try:
            from student.callbacks import get_student_callbacks as _get_callbacks
        except Exception:
            _get_callbacks = None

    # Prepare settings
    label_count = len(data.prepare_words_list(args.wanted_words.split(",")))
    model_settings = models.prepare_model_settings(
        label_count,
        args.sample_rate,
        args.clip_duration_ms,
        args.window_size_ms,
        args.window_stride_ms,
        args.dct_coefficient_count,
    )

    # Balance defaults for silence/unknown if missing
    num_classes = len(args.wanted_words.split(",")) + 2
    if args.silence_percentage is None:
        args.silence_percentage = 100.0 / num_classes
    if args.unknown_percentage is None:
        args.unknown_percentage = 100.0 / num_classes

    # Data pipeline
    audio_processor = data.AudioProcessor(
        data_url=args.data_url,
        data_dir=args.data_dir,
        silence_percentage=args.silence_percentage,
        unknown_percentage=args.unknown_percentage,
        wanted_words=args.wanted_words.split(","),
        validation_percentage=args.validation_percentage,
        testing_percentage=args.testing_percentage,
        model_settings=model_settings,
        micro=args.micro,
    )

    # Build model
    model = models.get_model(model_settings, args.model_architecture, model_name=args.model_name)

    # Optimizer & schedule
    training_steps_list = list(map(int, args.how_many_training_steps.split(",")))
    learning_rates_list = list(map(float, args.learning_rate.split(",")))
    lr_boundary_list = training_steps_list[:-1]
    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=lr_boundary_list, values=learning_rates_list
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    # Datasets
    AUTOTUNE = getattr(tf.data, "AUTOTUNE", getattr(tf.data, "experimental", tf.data).AUTOTUNE)
    train_data = audio_processor.get_data(
        audio_processor.Modes.TRAINING,
        0.8,    # background_frequency (sensible default)
        0.1,    # background_volume (sensible default)
        int((100.0 * args.sample_rate) / 1000),  # time_shift_ms=100 as samples
    ).repeat().batch(args.batch_size).prefetch(AUTOTUNE)

    val_data = audio_processor.get_data(audio_processor.Modes.VALIDATION).batch(args.batch_size).prefetch(AUTOTUNE)

    # Training length
    steps_total = int(np.sum(training_steps_list))
    epochs = int(np.ceil(steps_total / args.eval_step_interval))

    # Callbacks: best checkpoint + optional early stopping
    run_dir = Path(args.train_dir) / args.model_name / "best"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=(run_dir / (args.model_name + "_{val_accuracy:.3f}_ckpt")).as_posix(),
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    )
    callbacks = [ckpt_cb]
    if _get_callbacks:
        callbacks.extend(_get_callbacks())

    # Train
    model.fit(
        x=train_data,
        steps_per_epoch=args.eval_step_interval,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks,
    )

    # Evaluate on test set
    test_data = audio_processor.get_data(audio_processor.Modes.TESTING).batch(args.batch_size)
    test_loss, test_acc = model.evaluate(x=test_data)
    print(f"Final test accuracy: {test_acc*100:.2f}%")

    # Export a stable "best" checkpoint copy
    latest = tf.train.latest_checkpoint(run_dir)
    if latest:
        latest_name = Path(latest).name
        for src in Path(run_dir).parent.glob(f"best/{latest_name}.*"):
            dest = src.with_name(f"{args.model_name}_best_ckpt{src.suffix}")
            try:
                import shutil
                shutil.copy(src, dest)
            except Exception as e:
                print(f"Warning: could not copy {src} -> {dest}: {e}")
    print("✅ Training run finished.")

