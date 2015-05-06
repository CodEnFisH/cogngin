/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.dlps

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.mllib.linalg.Vector

/**
 * :: DeveloperApi ::
 * Class used to compute the gradient for a loss function, given a single data point.
 */
@DeveloperApi
abstract class PSGradient extends Serializable {
  /**
   * Compute the gradient and loss given the features of a single data point.
   *
   * @param data features for one data point
   * @param label label for this data point
   *
   * @return (gradient: Vector, loss: Double)
   */
  def compute(data: Vector, label: Double): Double

}

class PSSimpleGradient(nn: NeuralNetwork) extends PSGradient {
    val numIn = nn.numInput
    val numLabel = nn.numOut

    override def compute(data: Vector, label: Double): Double = {

      var input: BDM[Double] = null
      var label: BDM[Double] = null
      val batchedData = data.toArray
      if (data.size != numIn + numLabel) {
        val numCol = data.size / (numIn + numLabel)
        input = new BDM[Double](numIn, numCol, batchedData)
        label = new BDM[Double](numLabel, numCol, batchedData, numIn * numCol)
      }
      else {
        input = new BDV(batchedData, 0, 1, numIn).toDenseMatrix.t
        label = new BDV(batchedData, numIn, 1, numLabel).toDenseMatrix.t
      }

      // get the grads from cassandra.
      val error = nn.iterate(input, label)
      // put the get grads function here. Like the foldToVector thing.

      error
  }
}

/*
/**
 * :: DeveloperApi ::
 * Compute gradient and loss for a logistic loss function, as used in binary classification.
 * See also the documentation for the precise formulation.
 */
@DeveloperApi
class LogisticGradient extends PSGradient {
  override def compute(data: Vector, label: Double): (Vector, Double) = {
    val margin = -1.0 * dot(data, weights)
    val gradientMultiplier = (1.0 / (1.0 + math.exp(margin))) - label
    val gradient = data.copy
    scal(gradientMultiplier, gradient)
    val loss =
      if (label > 0) {
        math.log1p(math.exp(margin)) // log1p is log(1+p) but more accurate for small p
      } else {
        math.log1p(math.exp(margin)) - margin
      }

    (gradient, loss)
  }
}

/**
 * :: DeveloperApi ::
 * Compute gradient and loss for a Least-squared loss function, as used in linear regression.
 * This is correct for the averaged least squares loss function (mean squared error)
 *              L = 1/n ||A weights-y||^2
 * See also the documentation for the precise formulation.
 */
@DeveloperApi
class LeastSquaresGradient extends PSGradient {
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val diff = dot(data, weights) - label
    val loss = diff * diff
    val gradient = data.copy
    scal(2.0 * diff, gradient)
    (gradient, loss)
  }

  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {
    val diff = dot(data, weights) - label
    axpy(2.0 * diff, data, cumGradient)
    diff * diff
  }
}

/**
 * :: DeveloperApi ::
 * Compute gradient and loss for a Hinge loss function, as used in SVM binary classification.
 * See also the documentation for the precise formulation.
 * NOTE: This assumes that the labels are {0,1}
 */
@DeveloperApi
class HingeGradient extends PSGradient {
  override def compute(data: Vector, label: Double, weights: Vector): (Vector, Double) = {
    val dotProduct = dot(data, weights)
    // Our loss function with {0, 1} labels is max(0, 1 - (2y – 1) (f_w(x)))
    // Therefore the gradient is -(2y - 1)*x
    val labelScaled = 2 * label - 1.0
    if (1.0 > labelScaled * dotProduct) {
      val gradient = data.copy
      scal(-labelScaled, gradient)
      (gradient, 1.0 - labelScaled * dotProduct)
    } else {
      (Vectors.sparse(weights.size, Array.empty, Array.empty), 0.0)
    }
  }

  override def compute(
      data: Vector,
      label: Double,
      weights: Vector,
      cumGradient: Vector): Double = {
    val dotProduct = dot(data, weights)
    // Our loss function with {0, 1} labels is max(0, 1 - (2y – 1) (f_w(x)))
    // Therefore the gradient is -(2y - 1)*x
    val labelScaled = 2 * label - 1.0
    if (1.0 > labelScaled * dotProduct) {
      axpy(-labelScaled, data, cumGradient)
      1.0 - labelScaled * dotProduct
    } else {
      0.0
    }
  }
}
*/