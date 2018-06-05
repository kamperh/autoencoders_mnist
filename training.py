"""
Training functions.

Author: Herman Kamper
Date: 2016, 2018
"""

from __future__ import division
from __future__ import print_function
from datetime import datetime
import numpy as np
import sys
import tensorflow as tf
import timeit


def train_fixed_epochs(n_epochs, optimizer, train_loss_tensor,
        train_feed_iterator, feed_placeholders, validation_loss_tensor=None,
        validation_feed_iterator=None, load_model_fn=None, save_model_fn=None,
        save_best_val_model_fn=None, config=None, epoch_offset=0):
    """
    Train a model for a fixed number of epochs.
    
    Parameters
    ----------
    train_loss : Tensor
        The function that is optimized; should match the feed specified through
        `train_feed_iterator` and `feed_placeholders`. This can also be a list
        of Tensors, in which case the training loss is output as an array.
    train_feed_batch_iterator : generator
        Generates the values for the `feed_placeholders` for each training
        batch.
    feed_placeholders : list of placeholder
        The placeholders that is required for the `train_loss` (and optionally
        `validation_loss`) feeds.
    load_model_fn : str
        If provided, initialize session from this file.
    save_model_fn : str
        If provided, save final session to this file.
    save_best_val_model_fn : str
        If provided, save the best validation session to this file. If the
        `validation_loss_tensor` is a list of Tensors, the last value is taken
        as the current validation loss.
    
    Return
    ------
    record_dict : dict
        Statistics tracked during training. Each key describe the statistic,
        while the value is a list of (epoch, value) tuples.
    """

    assert save_best_val_model_fn is None or validation_loss_tensor is not None
    
    # Statistics
    record_dict = {}
    record_dict["epoch_time"] = []
    record_dict["train_loss"] = []
    if validation_loss_tensor is not None:
        record_dict["validation_loss"] = []
        best_validation_loss = np.inf
    
    print(datetime.now())
    
    def feed_dict(vals):
        return {key: val for key, val in zip(feed_placeholders, vals)}

    # Launch the graph
    saver = tf.train.Saver()
    if load_model_fn is None:
        init = tf.global_variables_initializer()
    with tf.Session(config=config) as session:
        
        # Start or restore session
        if load_model_fn is None:
            session.run(init)
        else:
            saver.restore(session, load_model_fn)
    
        # Train
        for i_epoch in xrange(n_epochs):
            print("Epoch {}:".format(epoch_offset + i_epoch)),
            start_time = timeit.default_timer()
            
            # Train model
            train_losses = []
            if not isinstance(train_loss_tensor, (list, tuple)):
                for cur_feed in train_feed_iterator:
                    _, cur_loss = session.run(
                        [optimizer, train_loss_tensor],
                        feed_dict=feed_dict(cur_feed)
                        )
                    train_losses.append(cur_loss)
                train_loss = np.mean(train_losses)
            else:
                for cur_feed in train_feed_iterator:
                    cur_loss = session.run(
                        [optimizer] + train_loss_tensor,
                        feed_dict=feed_dict(cur_feed)
                        )
                    cur_loss.pop(0)  # remove the optimizer
                    cur_loss = np.array(cur_loss)
                    train_losses.append(cur_loss)
                train_loss = np.mean(train_losses, axis=0)
            record_dict["train_loss"].append((i_epoch, train_loss))

            # Validation model
            if validation_loss_tensor is not None:
                validation_losses = []
                if not isinstance(validation_loss_tensor, (list, tuple)):
                    for cur_feed in validation_feed_iterator:
                        cur_loss = session.run(
                            [validation_loss_tensor],
                            feed_dict=feed_dict(cur_feed)
                            )
                        validation_losses.append(cur_loss)
                    validation_loss = np.mean(validation_losses)
                    cur_validation_loss = validation_loss
                else:
                    for cur_feed in validation_feed_iterator:
                        cur_loss = session.run(
                            validation_loss_tensor,
                            feed_dict=feed_dict(cur_feed)
                            )
                        cur_loss = np.array(cur_loss)
                        validation_losses.append(cur_loss)
                    validation_loss = np.mean(validation_losses, axis=0)
                    cur_validation_loss = validation_loss[-1]
                record_dict["validation_loss"].append(
                    (i_epoch, validation_loss)
                    )

            # Statistics
            end_time = timeit.default_timer()
            epoch_time = end_time - start_time
            record_dict["epoch_time"].append((i_epoch, epoch_time))
            
            log = "{:.3f} sec".format(epoch_time)
            log += ", train loss: " + str(train_loss)
            if validation_loss is not None:
                log += ", val loss: " + str(validation_loss)
            if (save_best_val_model_fn is not None and cur_validation_loss <
                    best_validation_loss):
                saver.save(session, save_best_val_model_fn)
                best_validation_loss = cur_validation_loss
                log += " *"
            print(log)
            sys.stdout.flush()

        if save_model_fn is not None:
            print("Writing: {}".format(save_model_fn))
            saver.save(session, save_model_fn)
            
    total_time = sum([i[1] for i in record_dict["epoch_time"]])
    print("Training time: {:.3f} min".format(total_time/60.))
    
    print(datetime.now())
    return record_dict
