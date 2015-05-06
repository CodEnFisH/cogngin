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

import breeze.linalg.{DenseMatrix => BDenseMatrix, DenseVector => BDenseVector}
import org.apache.spark.Logging
import org.apache.spark.mllib.dlps.Activation._
import org.apache.spark.util.Utils

/**
 *                    Xn    delta
 *                    ^^    \/
 * Wn      -->  *********************
 *              *    Fn(Xn-1, Wn)   *
 *              *                   *
 * dE/dWn  <--  *********************
 *                    ^^    \/
 *                   Xn-1   delta
 * */

/**
 * Representing a layer of module function that maps input to the output
 * The basic functionality is to run the back propagation algorithm to train the network
 * */
private[mllib] trait Layer extends Serializable {

  // Maxmin: Layer should only maintain its own info.
  //         we should have a Model/NN abstraction, to host the general info,
  //         e.g., total number of layers, number of total iterations,
  //         convergence error threshold, etc.
  //var nextModule: Layer = null

  // with parameter server, do not need to save the data in the layer
  val inNum: Int

  val outNum: Int

  def activation(in: BDenseMatrix[Double]): Unit

  def derivative(d: BDenseMatrix[Double], out: BDenseMatrix[Double]): Unit

  def forward(
      in: BDenseMatrix[Double],
      weight: BDenseMatrix[Double],
      bias: BDenseVector[Double]): BDenseMatrix[Double] = {
    // forward: activation(W * X + B)
    assert(in.rows == weight.cols)
    val out: BDenseMatrix[Double] = weight * in
    for (i <- 0 until out.cols) {
      out(::, i) :+= bias
    }
    activation(out)
    out
  }

  def backprop(
      in: BDenseMatrix[Double],
      delta: BDenseMatrix[Double]): (BDenseMatrix[Double], BDenseVector[Double]) = {
    val wG = delta * in.t
    val bG = BDenseVector.zeros[Double](outNum)
    for (i <- 0 until in.cols) {
      bG :+= delta(::, i)
    }
    (wG, bG)
  }

  def deltaOutput(
      out: BDenseMatrix[Double],
      trueValue: BDenseMatrix[Double]): BDenseMatrix[Double] = {
    val delta = out - trueValue
    derivative(delta, out)
    delta
  }

  def deltaHidden(
      topWeight: BDenseMatrix[Double],
      topDelta: BDenseMatrix[Double],
      out: BDenseMatrix[Double]) = {
    val delta = topWeight.t * topDelta
    derivative(delta, out)
    delta
  }
}

/*
 * Sigmoid Layer
 */

private[mllib] class SigmoidLayer(in: Int, out: Int) extends {
  val inNum = in
  val outNum = out
} with Layer with Logging {

  def activation(in: BDenseMatrix[Double]): Unit = {
    for (i <- 0 until in.rows) {
      for (j <- 0 until in.cols) {
        in(i, j) = sigmoid(in(i, j))
      }
    }
  }

  def derivative(delta: BDenseMatrix[Double], out: BDenseMatrix[Double]): Unit = {
    for (i <- 0 until delta.rows) {
      for (j <- 0 until delta.cols) {
        delta(i, j) = delta(i, j) * sigmoidPrime(out(i, j))
      }
    }
  }
}

/**
 * Tanh Layer
 */

private[mllib] class TanhLayer(in: Int, out: Int) extends {
  val inNum = in
  val outNum = out
} with Layer with Logging {

  def activation(in: BDenseMatrix[Double]): Unit = {
    for (i <- 0 until in.rows) {
      for (y <- 0 until in.cols) {
        in(i, y) = tanh(in(i, y))
      }
    }
  }

  def derivative(delta: BDenseMatrix[Double], out: BDenseMatrix[Double]): Unit = {
    for (i <- 0 until delta.rows) {
      for (j <- 0 until delta.cols) {
        delta(i, j) = delta(i, j) * tanhPrime(out(i, j))
      }
    }
  }
}

/**
 * Softmax Layer
 */

private[mllib] class SoftmaxLayer(in: Int, out: Int) extends {
  val inNum = in
  val outNum = out
} with Layer with Logging {

  def activation(in: BDenseMatrix[Double]): Unit = {
    for (col <- 0 until in.cols) {
      softmax(in(::, col))
    }
  }

  def derivative(
    in: BDenseMatrix[Double],
    output: BDenseMatrix[Double]): Unit = {
  }
}

private[mllib] object Layer {

  //tools and utils for the layer abstractions

  def initBias(outNum: Int): BDenseVector[Double] = {
    BDenseVector.zeros[Double](outNum)
  }

  // initialize weight to all zeros
  def initWeight(inNum: Int, outNum: Int): BDenseMatrix[Double] = {
     BDenseMatrix.zeros[Double](outNum, inNum)
  }

  // initialize weight with random number
  def initWeight(inNum: Int, outNum: Int, rand: () => Double): BDenseMatrix[Double] = {
    val w = initWeight(inNum, outNum)
    for(i <- 0 until w.data.length) {
      w.data(i) = rand()
    }
    w
  }

  // initialize weight to (-value~value)
  def initWeight(inNum: Int, outNum: Int, value: Double): BDenseMatrix[Double] = {
    val s = if(value <= 0) 4D * math.sqrt(6D / (inNum + outNum)) else value
    val w = initWeight(inNum, outNum)
    for(i <- 0 until w.data.length) {
      w.data(i) = (2D*Utils.random.nextDouble() - 1D) * value
    }
    w
  }
}

