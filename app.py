from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import librosa
import argparse
import torch
import sys
# from g2p.train import g2p
import soundfile as sf
from flask import Flask, render_template, url_for, redirect, request, send_file
import uuid
import os
import json

app = Flask(__name__)
vocoder_path = Path("vocoder/saved_models/pretrained/pretrained.pt")
synthesizer_path = Path("synthesizer/saved_models/logs-pretrained/")
encoder_path = Path("encoder/saved_models/pretrained.pt")


# print("Running a test of your configuration...\n")
# if not torch.cuda.is_available():
#     print("Your PyTorch installation is not configured to use CUDA. If you have a GPU ready "
#           "for deep learning, ensure that the drivers are properly installed, and that your "
#           "CUDA version matches your PyTorch installation. CPU-only inference is currently "
#           "not supported.", file=sys.stderr)
#     quit(-1)
device_id = torch.cuda.current_device()
gpu_properties = torch.cuda.get_device_properties(device_id)
print("Found %d GPUs available. Using GPU %d (%s) of compute capability %d.%d with "
      "%.1fGb total memory.\n" %
      (torch.cuda.device_count(),
       device_id,
       gpu_properties.name,
       gpu_properties.major,
       gpu_properties.minor,
       gpu_properties.total_memory / 1e9))

## Load the models one by one.
print("Preparing the encoder, the synthesizer and the vocoder...")
encoder.load_model(encoder_path)
synthesizer = Synthesizer(synthesizer_path.joinpath("taco_pretrained"), low_mem=False)
vocoder.load_model(vocoder_path)

# ## Run a test
# print("Testing your configuration with small inputs.")
# # Forward an audio waveform of zeroes that lasts 1 second. Notice how we can get the encoder's
# # sampling rate, which may differ.
# # If you're unfamiliar with digital audio, know that it is encoded as an array of floats
# # (or sometimes integers, but mostly floats in this projects) ranging from -1 to 1.
# # The sampling rate is the number of values (samples) recorded per second, it is set to
# # 16000 for the encoder. Creating an array of length <sampling_rate> will always correspond
# # to an audio of 1 second.
# print("\tTesting the encoder...")
# encoder.embed_utterance(np.zeros(encoder.sampling_rate))
#
# # Create a dummy embedding. You would normally use the embedding that encoder.embed_utterance
# # returns, but here we're going to make one ourselves just for the sake of showing that it's
# # possible.
# embed = np.random.rand(speaker_embedding_size)
# # Embeddings are L2-normalized (this isn't important here, but if you want to make your own
# # embeddings it will be).
# embed /= np.linalg.norm(embed)
# # The synthesizer can handle multiple inputs with batching. Let's create another embedding to
# # illustrate that
# embeds = [embed, np.zeros(speaker_embedding_size)]
# texts = ["test 1", "test 2"]
# print("\tTesting the synthesizer... (loading the model will output a lot of text)")
# mels = synthesizer.synthesize_spectrograms(texts, embeds)
#
# # The vocoder synthesizes one waveform at a time, but it's more efficient for long ones. We
# # can concatenate the mel spectrograms to a single one.
# mel = np.concatenate(mels, axis=1)
# # The vocoder can take a callback function to display the generation. More on that later. For
# # now we'll simply hide it like this:
# no_action = lambda *args: None
# print("\tTesting the vocoder...")
# # For the sake of making this test short, we'll pass a short target length. The target length
# # is the length of the wav segments that are processed in parallel. E.g. for audio sampled
# # at 16000 Hertz, a target length of 8000 means that the target audio will be cut in chunks of
# # 0.5 seconds which will all be generated together. The parameters here are absurdly short, and
# # that has a detrimental effect on the quality of the audio. The default parameters are
# # recommended in general.
# vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
#
# print("All test passed! You can now synthesize speech.\n\n")


@app.route('/<page>', methods=['GET', 'POST'])
def upload(page):
    if request.method == 'POST':
        # Set the uploaded file a uuid name
        filename_uuid = str(uuid.uuid4()) + '.wav'
        path_uuid = os.path.abspath(os.path.join(os.getcwd(), 'audio', filename_uuid))

        blob = request.files['file']
        text = request.form['text']

        # f = open(path_uuid, 'wb')
        # f.write(blob)
        # f.close()
        blob.save(path_uuid)

        in_fpath = path_uuid

        try:
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            # - If the wav is already loaded:
            original_wav, sampling_rate = librosa.load(str(in_fpath))
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            print("Loaded file succesfully")

            # Then we derive the embedding. There are many functions and parameters that the
            # speaker encoder interfaces. These are mostly for in-depth research. You will typically
            # only use this function (with its default parameters):
            embed = encoder.embed_utterance(preprocessed_wav)
            print("Created the embedding")

            ## Generating the spectrogram
            # text = input("Write a sentence (+-20 words) to be synthesized:\n")

            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            embeds = [embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")

            ## Generating the waveform
            print("Synthesizing the waveform:")

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(spec)

            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

            # Trim excess silences to compensate for gaps in spectrograms (issue #53)
            generated_wav = encoder.preprocess_wav(generated_wav)
            # Save it on the disk
            num_generated = 0
            filename = "demo_output_%02d.wav" % num_generated
            print(generated_wav.dtype)
            sf.write('static/output/' + filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            num_generated += 1
            print("\nSaved output as %s\n\n" % filename)
            return render_template(page + '.html', displayment='inline', file=url_for('static', filename='output/' + filename))

        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
            return render_template(page + '.html', displayment='none')
    return render_template(page + '.html', displayment='none')


@app.route('/recorder')
def recorder():
    return render_template('voice.html')


@app.route('/post/record', methods=['POST'])
def record():
    if request.method == 'POST':
        # Set the uploaded file a uuid name
        filename_uuid = str(uuid.uuid4()) + '.wav'
        path_uuid = os.path.abspath(os.path.join(os.getcwd(), 'audio', filename_uuid))

        blob = request.form['audio']
        text = request.form['text']

        f = open(path_uuid, 'wb')
        f.write(blob)
        f.close()
        blob.save(path_uuid)

        in_fpath = path_uuid

        try:
            preprocessed_wav = encoder.preprocess_wav(in_fpath)
            # - If the wav is already loaded:
            original_wav, sampling_rate = librosa.load(str(in_fpath))
            preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
            print("Loaded file succesfully")

            # Then we derive the embedding. There are many functions and parameters that the
            # speaker encoder interfaces. These are mostly for in-depth research. You will typically
            # only use this function (with its default parameters):
            embed = encoder.embed_utterance(preprocessed_wav)
            print("Created the embedding")

            ## Generating the spectrogram
            # text = input("Write a sentence (+-20 words) to be synthesized:\n")

            # The synthesizer works in batch, so you need to put your data in a list or numpy array
            texts = [text]
            embeds = [embed]
            # If you know what the attention layer alignments are, you can retrieve them here by
            # passing return_alignments=True
            specs = synthesizer.synthesize_spectrograms(texts, embeds)
            spec = specs[0]
            print("Created the mel spectrogram")

            ## Generating the waveform
            print("Synthesizing the waveform:")

            # Synthesizing the waveform is fairly straightforward. Remember that the longer the
            # spectrogram, the more time-efficient the vocoder.
            generated_wav = vocoder.infer_waveform(spec)

            ## Post-generation
            # There's a bug with sounddevice that makes the audio cut one second earlier, so we
            # pad it.
            generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")

            # Trim excess silences to compensate for gaps in spectrograms (issue #53)
            generated_wav = encoder.preprocess_wav(generated_wav)
            # Save it on the disk
            num_generated = 0
            filename = "demo_output_%02d.wav" % num_generated
            print(generated_wav.dtype)
            sf.write('static/output/' + filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
            num_generated += 1
            print("\nSaved output as %s\n\n" % filename)
            return url_for('static', filename='output/' + filename), 200

        except Exception as e:
            print("Caught exception: %s" % repr(e))
            print("Restarting\n")
            return 500
    return 200


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
