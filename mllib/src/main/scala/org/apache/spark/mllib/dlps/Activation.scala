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

import breeze.linalg.{DenseVector, max => bmax}

// TODO: add more functions
object Activation {
  /**
   * sigmoid function for vector
   * @param x
   * @return
   */
  def sigmoid(x: DenseVector[Double]): DenseVector[Double] = x.map(sigmoid)

  /**
   * sigmoid function for double value
   * @param input
   * @return
   */
  @inline def sigmoid(input: Double): Double = 1.0 / (1.0 + Math.exp(-input))

  /**
   * Derivative of sigmoid function for vector
   * @param x
   * @return
   */
  def sigmoidPrime(x: DenseVector[Double]): DenseVector[Double] = x.map(sigmoidPrime)

  /**
   * Derivative of sigmoid function for double value
   * @param input
   * @return
   */
  @inline def sigmoidPrime(input: Double): Double = sigmoid(input) * (1.0 - sigmoid(input))

  /**
   * Hyperbolic tangent for vector
   * @param x
   * @return
   */
  @inline def tanh(x: DenseVector[Double]): DenseVector[Double] = x.map(tanh)

  /**
   * Hyperbolic tangent for double value
   * @param input
   * @return
   */
  @inline def tanh(input: Double): Double = Math.tanh(input)

  /**
   * Derivative of hyperbolic tangent for vector
   * @param x
   * @return
   */
  def tanhPrime(x: DenseVector[Double]): DenseVector[Double] = x.map(tanhPrime)

  /**
   * Derivative of hyperbolic tangent for double value
   * @param input
   * @return
   */
  @inline def tanhPrime(input: Double): Double = 1.0 - Math.pow(tanh(input), 2.0)

  @inline def softmax(in: DenseVector[Double]): DenseVector[Double] = {
    val max = bmax(in)
    var denom = 0D
    for(i <- 0 until in.length) {
      in(i) = Math.exp(in(i) - max)
      denom += in(i)
    }
    in :/= denom
  }
}
