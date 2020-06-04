#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import gpt2.model as model
import gpt2.sample as sample
import gpt2.encoder as encoder

def interact_model(
    model_path='gpt2/models',
    model_name='117M',
    raw_text='== Article Start ==\n',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=0.0
):
    """
    Interactively run the model

    :model_path=gpt2/models : String, where the model folder is
    :model_name=117M : String, which model to use
    :raw_text='== Article Start ==\n' : Context to start the generated sample/s.
     The default is the string to start a new article from the beginning. To start
     an article with a custom title, use the default text then the desired title,
     for example, '== Article Start ==\nWorld War II'. If the default text is not
     prepended the model will likely not start at the beginning of an article.
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Maximum number of tokens in generated text, if None (default), 
     is a maximum of 1024
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    enc = encoder.get_encoder(model_path, model_name)
    hparams = model.default_hparams()
    with open(os.path.join(model_path, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(model_path, model_name))
        saver.restore(sess, ckpt)

        context_tokens = enc.encode(raw_text)
        generated = 0
        samples = []
       
        while generated < nsamples:
            # e.g. if we have batch_size=3, want 20 samples, have 19, only
            #n_gen = min(batch_size, nsamples-generated)
            # previous line doesn't work, some other code is disrupted if 
            # batch_size is reduced
            n_gen = batch_size 
          
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(n_gen)]
            })[:, len(context_tokens):]
            
            for i in range(n_gen):             
                text = enc.decode(out[i])
                
                # rules to reject a sample
                rules = [# is it a complete article
                        '<|endoftext|>' in text, 
                         # don't want it too short
                         len(text.split()) > 50, 
                         # does it do title -> intro break correctly
                         text.find('\n')==text.find('\n') 
                         ]
                # other possible rejection rules:
                # 'list of' in title
                # 2 or more consecutive empty sections that aren't one of:
                  # references, see also, external links, citations
                # 
                

                if all(rules): 
                  endIndex = text.find('<|endoftext|>')
                  samples.append(text[:endIndex])
                  generated += 1 

    return samples
 
if __name__ == '__main__':
    fire.Fire(interact_model)








