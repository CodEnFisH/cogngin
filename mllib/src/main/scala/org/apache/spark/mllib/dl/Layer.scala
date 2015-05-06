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

package org.apache.spark.mllib.dl

import breeze.linalg.{DenseMatrix => DMatrix, DenseVector => DVector}
import org.apache.spark.Logging

import Activation._
import Layer._
import org.apache.spark.util.Utils

import scala.util.Random

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
  def bias: DVector[Double]

  def weight: DMatrix[Double]

  def inNum = weight.cols

  def outNum = weight.rows

  protected lazy val rand: Random = new Random()

  def setRandSeed(seed: Long) = {
    rand.setSeed(seed)
  }

  def setBias(b: DVector[Double])

  def setWeight(w: DMatrix[Double])

  def activation(in: DMatrix[Double]): Unit

  def primitive(d: DMatrix[Double], out: DMatrix[Double]): Unit

  def forward(in: DMatrix[Double]): DMatrix[Double] = {
    // forward: activation(W * X + B)
    assert(in.rows == weight.cols)
    val out: DMatrix[Double] = weight * in
    for (i <- 0 until out.cols) {
      out(::, i) :+= bias
    }
    activation(out)
    out
  }

  def backprop(in: DMatrix[Double], delta: DMatrix[Double]): (DMatrix[Double], DVector[Double]) = {
    val wG = delta * in.t
    val bG = DVector.zeros[Double](outNum)
    for (i <- 0 until in.cols) {
      bG :+= delta(::, i)
    }
    (wG, bG)
  }

  def deltaOutput(out: DMatrix[Double], trueValue: DMatrix[Double]): DMatrix[Double] = {
    val delta = out - trueValue
    primitive(delta, out)
    delta
  }

  def deltaHidden(topLayer: Layer, topDelta: DMatrix[Double], out: DMatrix[Double]) = {
    val delta = topLayer.weight.t * topDelta
    primitive(delta, out)
    delta
  }
}

/*
 * Sigmoid Layer
 */

private[mllib] class SigmoidLayer(
  var weight: DMatrix[Double] = null,
  var bias: DVector[Double] = null) extends Layer with Logging {

  def this(inNum: Int, outNum: Int) {
    this(initWeight(inNum, outNum, 4D * math.sqrt(6D / (inNum + outNum))),
      initBias(outNum))
  }

  def setBias(bias: DVector[Double]): Unit = {
    this.bias = bias
  }

  def setWeight(weight: DMatrix[Double]): Unit = {
    this.weight = weight
  }

  def activation(in: DMatrix[Double]): Unit = {
    for (i <- 0 until in.rows) {
      for (j <- 0 until in.cols) {
        in(i, j) = sigmoid(in(i, j))
      }
    }
  }

  def primitive(delta: DMatrix[Double], out: DMatrix[Double]): Unit = {
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

private[mllib] class TanhLayer(
  var weight: DMatrix[Double] = null,
  var bias: DVector[Double] = null) extends Layer with Logging {

  def this(numIn: Int, numOut: Int) {
    this(initWeight(numIn, numOut, math.sqrt(6D / (numIn + numOut))),
      initBias(numOut))
  }

  def setBias(bias: DVector[Double]): Unit = {
    this.bias = bias
  }

  def setWeight(weight: DMatrix[Double]): Unit = {
    this.weight = weight
  }

  def activation(in: DMatrix[Double]): Unit = {
    for (i <- 0 until in.rows) {
      for (y <- 0 until in.cols) {
        in(i, y) = tanh(in(i, y))
      }
    }
  }

  def primitive(delta: DMatrix[Double], out: DMatrix[Double]): Unit = {
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

private[mllib] class SoftmaxLayer(
  var weight: DMatrix[Double] = null,
  var bias: DVector[Double] = null) extends Layer with Logging {

  def this(inNum: Int, outNum: Int) {
    this(initWeight(inNum, outNum), initBias(outNum))
  }

  def setBias(bias: DVector[Double]): Unit = {
    this.bias = bias
  }

  def setWeight(weight: DMatrix[Double]): Unit = {
    this.weight = weight
  }

  def activation(in: DMatrix[Double]): Unit = {
    for (col <- 0 until in.cols) {
      softmax(in(::, col))
    }
  }

  def primitive(
    in: DMatrix[Double],
    output: DMatrix[Double]): Unit = {
  }
}

private[mllib] object Layer {

  //tools and utils for the layer abstractions

  def initBias(outNum: Int): DVector[Double] = {
    DVector.zeros[Double](outNum)
  }

  // initialize weight to all zeros
  def initWeight(inNum: Int, outNum: Int): DMatrix[Double] = {
     DMatrix.zeros[Double](outNum, inNum)
  }

  // initialize weight with random number
  def initWeight(inNum: Int, outNum: Int, rand: () => Double): DMatrix[Double] = {
    val w = initWeight(inNum, outNum)
    for(i <- 0 until w.data.length) {
      w.data(i) = rand()
    }
    w
  }

  // initialize weight to (-value~value)
  def initWeight(inNum: Int, outNum: Int, value: Double): DMatrix[Double] = {
    val s = if(value <= 0) 4D * math.sqrt(6D / (inNum + outNum)) else value
    val w = initWeight(inNum, outNum)
    for(i <- 0 until w.data.length) {
      w.data(i) = (2D*Utils.random.nextDouble() - 1D) * value
    }
    w
  }
}

