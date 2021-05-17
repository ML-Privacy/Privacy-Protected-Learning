from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import ops
from tensorflow.python.training.optimizer import *
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops, state_ops
from tensorflow.python.training import training_ops
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import parameter_server_strategy

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend

import tensorflow as tf
import tensorflow_probability as tfp
import time
import numpy

import functools

#from keras.optimizers import Optimizer, SGD
import keras.backend as K
#from keras.models import model_from_json
#from keras.utils.generic_utils import CustomObjectScope

from Params import Params


class QCDGD(Optimizer):


    _HAS_AGGREGATE_GRAD = True

    first = True

    def __init__(self,
                learning_rate=0.01,
                momentum=0.0,
                nb_agents=5,
                params=None,
                nesterov=False,
                name="QCDGD",
                clip=0,
                ternSt=0,
                c1=0.5,
                delta=0.48,
                **kwargs):
   
        super(QCDGD, self).__init__(False, name)
        #self._set_hyper("learning_rate", kwargs.get("lr", 1))
        #self._set_hyper("decay", self._initial_decay)

        self._momentum = False
        if isinstance(momentum, ops.Tensor) or callable(momentum) or momentum > 0:
            self._momentum = True

        if isinstance(momentum, (int, float)) and (momentum < 0 or momentum > 1):
            raise ValueError("`momentum` must be between [0, 1].")

        #self._set_hyper("momentum", momentum)

        self.nesterov = nesterov
        self.learning_rate = learning_rate
        self.nb_agents = nb_agents
        
        if params == None:
            self.params = Params(nb_agents, 1)
        else:
            self.params = params

        self.epochStart = True

        self.clipSTD = clip
        self.stMultiplier = ternSt

        self.c1 = c1
        self.delta = delta

    def _create_slots(self, var_list):
        for var in var_list:
            self._zeros_slot(var, "epoch_var", self._name)
        if self._momentum:
            for var in var_list:
                self._zeros_slot(var, "momentum", self._name)

    def get_slot(self, var, name):
        return self._get_or_make_slot(var, var, name, self._name)
    
    def set_slot(self, var, name, val):
        m = self._get_or_make_slot(var, None, name, self._name)
        return m.assign(val)
    
    def clip(self, var, c):
        if c == 0:
            return var

        std = tf.math.reduce_std(var)
        return tf.clip_by_value(var, -c*std,c*std, name=None)
    
    def getST(self, var):
        return tf.math.reduce_max(tf.math.abs(var))
    
    def tern(self, var, st):
        if self.stMultiplier == 0:
            return var

        bernoulli = tf.math.abs(var) / st
        dist = tfp.distributions.Bernoulli(probs=bernoulli, dtype=tf.float32)
        bt = dist.sample()
        return tf.math.multiply(tf.math.sign(var), bt) * st

    # def st_tot()
    def CDGrads(self, converted_grads_and_vars, step):
        
        grad_list, var_list, processor = zip(*converted_grads_and_vars)

        layers = int(len(var_list) / (2 * self.nb_agents))
        
        val_list = [None] * len(grad_list)
        comb_list = [None] * len(grad_list)
        tern_list = [None] * len(grad_list)
        reg_list = [None] * len(grad_list)

        pi = self.params.genPi()
        bi = self.params.genBi() # Randomly generated column sch matrix
        # bi = self.params.genDiag() # Diagonal Matrix

        rand_s = self.params.genRand() # Gen single value since gradient is list
        # tf.print(rand_s)
      
        grad_list = list(grad_list)

        # floor_v = 0.0005
        # floor_v = .0005
        # floor_v = 0.001
        floor_v = -1

        # Norm = 0.5
        numerator = .5 
        e_decimal = 0.7

        epsilon = numerator / tf.math.pow(tf.cast(step + 1, tf.float32) * self.c1 + 1, e_decimal)
        # epsilon = tf.math.maximum(self.c1 / tf.math.pow(tf.cast(step + 1, tf.float32), self.delta), floor_v)
        # epsilon = self.c1
        # tf.print(epsilon)

        # Usually 1 before term
        # rand_s = numpy.random.random_sample(grad_list[2*(i + self.nb_agents * k)].get_shape().as_list())
        # lamb1 = .5 / tf.math.pow(tf.cast(step, tf.float32) * self.c1 + 1, 0.3) * numpy.ones(grad_list[2*(i + self.nb_agents * k)].get_shape().as_list()) + (rand_s / tf.math.maximum(tf.math.pow(tf.cast(step + 1, tf.float32), 2), floor_v))

        # rand_s = numpy.random.random_sample(grad_list[2*(i + self.nb_agents * k) + 1].get_shape().as_list())
        # lamb2 = .5 / tf.math.pow(tf.cast(step, tf.float32) * self.c1 + 1, 0.3) * numpy.ones(grad_list[2*(i + self.nb_agents * k)+1].get_shape().as_list())+ (rand_s / tf.math.maximum(tf.math.pow(tf.cast(step + 1, tf.float32), 2), floor_v))
        
        lamb1 = tf.math.maximum(numerator / tf.math.pow(tf.cast(step, tf.float32) * self.c1 + 1, 1-e_decimal), floor_v) 
        lamb2 = tf.math.maximum(numerator / tf.math.pow(tf.cast(step, tf.float32) * self.c1 + 1, 1-e_decimal), floor_v)

        for k in range(layers):
            
            st = tf.constant([0], dtype=tf.float32)
            stb = tf.constant([0], dtype=tf.float32)

            for i in range(self.nb_agents):
                for j in range(self.nb_agents):
                    # rand_s = numpy.random.random_sample(grad_list[2*(i + self.nb_agents * k)].get_shape().as_list())
                    # lamb1 = tf.math.maximum(numerator / tf.math.pow(tf.cast(step, tf.float32) * self.c1 + 1, 1-e_decimal), floor_v) * numpy.ones(grad_list[2*(i + self.nb_agents * k)].get_shape().as_list()) #+ (rand_s / tf.math.maximum(tf.math.pow(tf.cast(step + 1, tf.float32), 2), floor_v))

                    # rand_s = numpy.random.random_sample(grad_list[2*(i + self.nb_agents * k) + 1].get_shape().as_list())
                    # lamb2 = tf.math.maximum(numerator / tf.math.pow(tf.cast(step, tf.float32) * self.c1 + 1, 1-e_decimal), floor_v) * numpy.ones(grad_list[2*(i + self.nb_agents * k)+1].get_shape().as_list()) #+ (rand_s / tf.math.maximum(tf.math.pow(tf.cast(step + 1, tf.float32), 2), floor_v))
                        
                    if i != j:
                        tern_var = pi[i, j] * var_list[2*(j + self.nb_agents * k)] - bi[i, j] * lamb1 * grad_list[2*(j + self.nb_agents * k)]
                        tern_varb = pi[i, j] * var_list[2*(j + self.nb_agents * k) + 1] - bi[i, j] * lamb2 * grad_list[2*(j + self.nb_agents * k) + 1]

                        clip_var = self.clip(tern_var, self.clipSTD)
                        clip_varb = self.clip(tern_varb, self.clipSTD)

                        st = tf.math.maximum(st, self.getST(clip_var))
                        stb = tf.math.maximum(stb, self.getST(clip_varb))


                # st[k, i] = st_t * self.stMultiplier
                # stb_t[k, i] = stb_t * self.stMultiplier


            for i in range(self.nb_agents):
                newVal = var_list[2*(i + self.nb_agents * k)] * (1 - epsilon)
                newValB = var_list[2*(i + self.nb_agents * k) + 1] * (1 - epsilon)

                # Quant initial
                # newVal = var_list[2*(i + self.nb_agents * k)]
                # newValB = var_list[2*(i + self.nb_agents * k) + 1]

                # st = tf.constant([0], dtype=tf.float32)
                # stb = tf.constant([0], dtype=tf.float32)

                for j in range(self.nb_agents):

                    # Usually 1 before term
                    # rand_s = numpy.random.random_sample(grad_list[2*(i + self.nb_agents * k)].get_shape().as_list())
                    # lamb1 = tf.math.maximum(1 / tf.math.pow(tf.cast(step + 1, tf.float32), 1), floor_v) * numpy.ones(grad_list[2*(i + self.nb_agents * k)].get_shape().as_list()) + (rand_s / tf.math.maximum(tf.math.pow(tf.cast(step + 1, tf.float32), 2), floor_v))
                    # lamb1 = tf.math.maximum(numerator / tf.math.pow(tf.cast(step, tf.float32) * self.c1 + 1, 1-e_decimal), floor_v) * numpy.ones(grad_list[2*(i + self.nb_agents * k)].get_shape().as_list()) #+ (rand_s / tf.math.maximum(tf.math.pow(tf.cast(step + 1, tf.float32), 2), floor_v))

                    # rand_s = numpy.random.random_sample(grad_list[2*(i + self.nb_agents * k) + 1].get_shape().as_list())
                    # lamb2 = tf.math.maximum(1 / tf.math.pow(tf.cast(step + 1, tf.float32), 1), floor_v) * numpy.ones(grad_list[2*(i + self.nb_agents * k)+1].get_shape().as_list())+ (rand_s / tf.math.maximum(tf.math.pow(tf.cast(step + 1, tf.float32), 2), floor_v))
                    # lamb2 = tf.math.maximum(numerator / tf.math.pow(tf.cast(step, tf.float32) * self.c1 + 1, 1-e_decimal), floor_v) * numpy.ones(grad_list[2*(i + self.nb_agents * k)+1].get_shape().as_list()) #+ (rand_s / tf.math.maximum(tf.math.pow(tf.cast(step + 1, tf.float32), 2), floor_v))

                    if i == j:
                        newVal = newVal + epsilon * pi[i, j] * var_list[2*(i + self.nb_agents * k)] -  bi[i, j] * lamb1 * grad_list[2*(i + self.nb_agents * k)] * epsilon
                        newValB = newValB + epsilon * pi[i, j] * var_list[2*(i + self.nb_agents * k) + 1] -  bi[i, j] * lamb2 * grad_list[2*(i + self.nb_agents * k) + 1] * epsilon

                        # Quantized initial part
                        # tern_var = (-1 + pi[i, j]) * var_list[2*(i + self.nb_agents * k)] - bi[i, j] * lamb1 * grad_list[2*(i + self.nb_agents * k)]
                        # tern_varb = (-1 + pi[i, j]) * var_list[2*(i + self.nb_agents * k) + 1] - bi[i, j] * lamb2 * grad_list[2*(i + self.nb_agents * k) + 1]

                        # clip_var = self.clip(tern_var, self.clipSTD)
                        # clip_varb = self.clip(tern_varb, self.clipSTD)

                        # st = tf.math.maximum(st, self.getST(clip_var))
                        # stb = tf.math.maximum(stb, self.getST(clip_varb))
                        # # tf.print(st)

                        # q_v = self.tern(clip_var, st)
                        # q_vb = self.tern(clip_varb, stb)

                        # newVal = newVal + epsilon * q_v
                        # newValB = newValB + epsilon * q_vb
                        
                    else:

                        tern_var = pi[i, j] * var_list[2*(j + self.nb_agents * k)] - bi[i, j] * lamb1 * grad_list[2*(j + self.nb_agents * k)]
                        tern_varb = pi[i, j] * var_list[2*(j + self.nb_agents * k) + 1] - bi[i, j] * lamb2 * grad_list[2*(j + self.nb_agents * k) + 1]

                        clip_var = self.clip(tern_var, self.clipSTD)
                        clip_varb = self.clip(tern_varb, self.clipSTD)

                        # st = tf.math.maximum(st, self.getST(clip_var))
                        # stb = tf.math.maximum(stb, self.getST(clip_varb))
                        # tf.print(st)

                        # st = st * self.stMultiplier
                        # stb = stb * self.stMultiplier

                        q_v = self.tern(clip_var, st * self.stMultiplier)
                        q_vb = self.tern(clip_varb, stb * self.stMultiplier)

                        newVal = newVal + epsilon * q_v
                        newValB = newValB + epsilon * q_vb

                val_list[2*(i + self.nb_agents * k)] = newVal
                val_list[2*(i + self.nb_agents * k) + 1] = newValB


        return zip(grad_list, var_list, processor, val_list)


    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """Apply gradients to variables.

        This is the second part of `minimize()`. It returns an `Operation` that
        applies gradients.

        Args:
          grads_and_vars: List of (gradient, variable) pairs as returned by
            `compute_gradients()`.
          global_step: Optional `Variable` to increment by one after the
            variables have been updated.
          name: Optional name for the returned operation.  Default to the
            name passed to the `Optimizer` constructor.

        Returns:
          An `Operation` that applies the specified gradients. If `global_step`
          was not None, that operation also increments `global_step`.

        Raises:
          TypeError: If `grads_and_vars` is malformed.
          ValueError: If none of the variables have gradients.
          RuntimeError: If you should use `_distributed_apply()` instead.
        """
        # This is a default implementation of apply_gradients() that can be shared
        # by most optimizers.  It relies on the subclass implementing the following
        # methods: _create_slots(), _prepare(), _apply_dense(), and _apply_sparse().

        # TODO(isaprykin): Get rid of `has_strategy()` check by
        # always calling _distributed_apply(), using the default distribution
        # as needed.

        self.epochStart = self.params.epochStart
        self.params.epochStart = False

        if distribute_ctx.has_strategy():
          # Handle DistributionStrategy case.
          if distribute_ctx.in_cross_replica_context():
            raise RuntimeError("Use `_distributed_apply()` instead of "
                               "`apply_gradients()` in a cross-replica context.")

          grads_and_vars = get_filtered_grad_fn(lambda: grads_and_vars)()
          return distribute_ctx.get_replica_context().merge_call(
              self._distributed_apply, args=(grads_and_vars, global_step, name))

        # No DistributionStrategy case.
        grads_and_vars = tuple(grads_and_vars)  # Make sure repeat iteration works.
        if not grads_and_vars:
          raise ValueError("No variables provided.")
        converted_grads_and_vars = []

        

        for g, v in grads_and_vars:
          if g is not None:
            try:
              # Convert the grad to Tensor or IndexedSlices if necessary.
              g = ops.convert_to_tensor_or_indexed_slices(g)
            except TypeError:
              raise TypeError(
                  "Gradient must be convertible to a Tensor"
                  " or IndexedSlices, or None: %s" % g)
            if not isinstance(g, (ops.Tensor, ops.IndexedSlices)):
              raise TypeError(
                  "Gradient must be a Tensor, IndexedSlices, or None: %s" % g)
          p = _get_processor(v)
          converted_grads_and_vars.append((g, v, p))

        converted_grads_and_vars = tuple(converted_grads_and_vars)
        grad_list = [g for g, v, _ in converted_grads_and_vars if g is not None]
        var_list = [v for g, v, _ in converted_grads_and_vars if g is not None]
        if not var_list:
          raise ValueError("No gradients provided for any variable: %s." %
                           ([str(v) for _, v, _ in converted_grads_and_vars],))
        with ops.init_scope():
          self._create_slots(var_list)
        
        # Imp function
        compGV = self.CDGrads(converted_grads_and_vars, global_step)

        update_ops = []
        with ops.name_scope(name, self._name, skip_on_eager=False) as name:
          self._prepare()

          for grad, var, processor, val in compGV:
            if grad is None:
              continue
            # We colocate all ops created in _apply_dense or _apply_sparse
            # on the same device as the variable.
            # TODO(apassos): figure out how to get the variable name here.
            if (context.executing_eagerly() or
                resource_variable_ops.is_resource_variable(var)
                and not var._in_graph_mode):  # pylint: disable=protected-access
              scope_name = ""
            else:
              scope_name = var.op.name
            with ops.name_scope(
                "update_" + scope_name,
                skip_on_eager=False), ops.colocate_with(var):

              if self.epochStart and False:
                update_ops.append(self.set_slot(var, "epoch_var", val))
              
             
              update_ops.append(var.assign(val))
              

            
          
          if global_step is None:
            apply_updates = self._finish(update_ops, name)
          else:
            with ops.control_dependencies([self._finish(update_ops, "update")]):
              with ops.colocate_with(global_step):
                if isinstance(global_step, resource_variable_ops.BaseResourceVariable):
                  
                  apply_updates = resource_variable_ops.assign_add_variable_op( global_step.handle,
                                                                                ops.convert_to_tensor(1, dtype=global_step.dtype),
                                                                                name=name)
                else:
                  apply_updates = state_ops.assign_add(global_step, 1, name=name)

          if not context.executing_eagerly():
            if isinstance(apply_updates, ops.Tensor):
              apply_updates = apply_updates.op
            train_op = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
            if apply_updates not in train_op:
              train_op.append(apply_updates)

          if self.epochStart:
            self.epochStart = False
          return apply_updates

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype

        return training_ops.resource_apply_gradient_descent(
                var.handle, 1.0, grad, use_locking=self._use_locking)


    def get_config(self):
        config = super(SGD, self).get_config()
        config.update({
            "learning_rate": self._serialize_hyperparameter("learning_rate"),
            "decay": self._serialize_hyperparameter("decay"),
            "momentum": self._serialize_hyperparameter("momentum"),
            "nesterov": self.nesterov,
        })
        return config



def _filter_grads(grads_and_vars):
    """Filter out iterable with grad equal to None."""
    grads_and_vars = tuple(grads_and_vars)
    if not grads_and_vars:
        return grads_and_vars
    filtered = []
    vars_with_empty_grads = []
    for grad, var in grads_and_vars:
        if grad is None:
            vars_with_empty_grads.append(var)    
        else:
            filtered.append((grad, var))
    filtered = tuple(filtered)
    if not filtered:
        raise ValueError("No gradients provided for any variable: %s." %
                        ([v.name for _, v in grads_and_vars],))
    if vars_with_empty_grads:
        logging.warning(
            ("Gradients do not exist for variables %s when minimizing the loss."),
            ([v.name for v in vars_with_empty_grads]))
    return filtered











































import abc

import six

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribution_strategy_context as distribute_ctx
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import slot_creator
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


def get_filtered_grad_fn(grad_fn):
  # `distributed_context.join()` requires that its arguments are parallel
  # across threads, and in particular that `grads_and_vars` has the same
  # variables in the same order.

  # When computing gradients in eager mode with multiple threads, you
  # can get extra variables with a gradient of `None`. This happens when
  # those variables are accessed in another thread during the gradient
  # computation. To get a consistent set of variables, we filter out
  # those with `None` gradients.
  def filtered_grad_fn(*args, **kwargs):
    return [(g, v) for g, v in grad_fn(*args, **kwargs) if g is not None]

  return filtered_grad_fn


def _deduplicate_indexed_slices(values, indices):
  """Sums `values` associated with any non-unique `indices`.

  Args:
    values: A `Tensor` with rank >= 1.
    indices: A one-dimensional integer `Tensor`, indexing into the first
      dimension of `values` (as in an IndexedSlices object).
  Returns:
    A tuple of (`summed_values`, `unique_indices`) where `unique_indices` is a
    de-duplicated version of `indices` and `summed_values` contains the sum of
    `values` slices associated with each unique index.
  """
  unique_indices, new_index_positions = array_ops.unique(indices)
  summed_values = math_ops.unsorted_segment_sum(
      values, new_index_positions,
      array_ops.shape(unique_indices)[0])
  return (summed_values, unique_indices)


def _var_key(var):
  # TODO(ashankar): Consolidate handling for eager and graph
  if hasattr(var, "op"):
    return (var.op.graph, var.op.name)
  return var._unique_id  # pylint: disable=protected-access


@six.add_metaclass(abc.ABCMeta)
class _OptimizableVariable(object):
  """Interface for abstracting over variables in the optimizers."""

  @abc.abstractmethod
  def target(self):
    """Returns the optimization target for this variable."""
    raise NotImplementedError("Calling an abstract method.")

  @abc.abstractmethod
  def update_op(self, optimizer, g):
    """Returns the update ops for updating the variable."""
    raise NotImplementedError("Calling an abstract method.")


class _RefVariableProcessor(_OptimizableVariable):
  """Processor for Variable."""

  def __init__(self, v):
    self._v = v

  def __str__(self):
    return "<_RefVariableProcessor(%s)>" % self._v

  def target(self):
    return self._v._ref()  # pylint: disable=protected-access

  def update_op(self, optimizer, g):
    if isinstance(g, ops.Tensor):
      update_op = optimizer._apply_dense(g, self._v)  # pylint: disable=protected-access
      if self._v.constraint is not None:
        with ops.control_dependencies([update_op]):
          return self._v.assign(self._v.constraint(self._v))
      else:
        return update_op
    else:
      assert isinstance(g, ops.IndexedSlices), ("Gradient ", g, " is neither a "
                                                "tensor nor IndexedSlices.")
      if self._v.constraint is not None:
        raise RuntimeError(
            "Cannot use a constraint function on a sparse variable.")
      # pylint: disable=protected-access
      return optimizer._apply_sparse_duplicate_indices(g, self._v)


class _DenseReadResourceVariableProcessor(_OptimizableVariable):
  """Processor for dense ResourceVariables."""

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v

  def update_op(self, optimizer, g):
    # pylint: disable=protected-access
    update_op = optimizer._resource_apply_dense(g, self._v.op.inputs[0])
    if self._v.constraint is not None:
      with ops.control_dependencies([update_op]):
        return self._v.assign(self._v.constraint(self._v))
    else:
      return update_op


class _DenseResourceVariableProcessor(_OptimizableVariable):
  """Processor for dense ResourceVariables."""

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v

  def update_op(self, optimizer, g):
    # pylint: disable=protected-access
    if isinstance(g, ops.IndexedSlices):
      if self._v.constraint is not None:
        raise RuntimeError(
            "Cannot use a constraint function on a sparse variable.")
      return optimizer._resource_apply_sparse_duplicate_indices(
          g.values, self._v, g.indices)
    update_op = optimizer._resource_apply_dense(g, self._v)
    if self._v.constraint is not None:
      with ops.control_dependencies([update_op]):
        return self._v.assign(self._v.constraint(self._v))
    else:
      return update_op


class _TensorProcessor(_OptimizableVariable):
  """Processor for ordinary Tensors.

  Even though a Tensor can't really be updated, sometimes it is useful to
  compute the gradients with respect to a Tensor using the optimizer. Updating
  the Tensor is, of course, unsupported.
  """

  def __init__(self, v):
    self._v = v

  def target(self):
    return self._v

  def update_op(self, optimizer, g):
    raise NotImplementedError("Trying to update a Tensor ", self._v)


def _get_processor(v):
  """The processor of v."""
  if context.executing_eagerly():
    if isinstance(v, ops.Tensor):
      return _TensorProcessor(v)
    else:
      return _DenseResourceVariableProcessor(v)
  if resource_variable_ops.is_resource_variable(v) and not v._in_graph_mode:  # pylint: disable=protected-access
    # True if and only if `v` was initialized eagerly.
    return _DenseResourceVariableProcessor(v)
  if v.op.type == "VarHandleOp":
    return _DenseResourceVariableProcessor(v)
  if isinstance(v, variables.Variable):
    return _RefVariableProcessor(v)
  if isinstance(v, ops.Tensor):
    return _TensorProcessor(v)
  raise NotImplementedError("Trying to optimize unsupported type ", v)
