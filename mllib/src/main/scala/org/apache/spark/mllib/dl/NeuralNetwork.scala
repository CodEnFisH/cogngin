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

import java.util.Random

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV}
import org.apache.spark.Logging

/**
 * NeuralNetwork representation
 * Consists of Layers.
 * Contains basic info for the NeuralNetwork
 *
 * Ln ----> Output
 * ...
 * L1    ----> Hidden Layer 1
 * L0    ----> Input
 *
 *
 */
class NeuralNetwork(val innerLayers: Array[Layer]) extends Logging with Serializable {

  /**
   * update the model with the delta value
   */
  /* TODO: make the NeuralNetwork serializable and also
           the layers that would be included in the neural network.
           Make them as a field in the training function, that would take sampleRDD
           and map to model update RDD.
   */

  val topology: Array[Int] = {
    val topology = new Array[Int](numLayer + 1)
    topology(0) = numInput
    for (i <- 1 to numLayer) {
      topology(i) = innerLayers(i - 1).outNum
    }
    topology
  }

  private lazy val rand: Random = new Random()

  def numLayer = innerLayers.length

  def numInput = innerLayers.head.inNum

  def numOut = innerLayers.last.outNum

  def predict(x: BDM[Double]): BDM[Double] = {
    var output = x
    for (layer <- 0 until numLayer) {
      output = innerLayers(layer).forward(output)
    }
    output
  }

  /**
   *
   * @param x, minibatch of sample vectors, each column is a vector
   * @param label, the true label
   * @return grads
   */
  private[mllib] def iterate(x: BDM[Double], label: BDM[Double]): (Array[(BDM[Double],
    BDV[Double])], Double, Double) = {
    val batchSize = x.cols
    val in = new Array[BDM[Double]](numLayer)
    val out = new Array[BDM[Double]](numLayer)
    val delta = new Array[BDM[Double]](numLayer)
    val grads = new Array[(BDM[Double], BDV[Double])](numLayer)

    for (layer <- 0 until numLayer) {
      val input = if (layer == 0) {
        x
      } else {
        out(layer - 1)
      }
      in(layer) = input

      val output = innerLayers(layer).forward(input)
      out(layer) = output
    }

    for (layer <- (0 until numLayer).reverse) {
      val input = in(layer)
      val output = out(layer)
      delta(layer) = {
        if (layer == numLayer - 1) {
          innerLayers(layer).deltaOutput(output, label)
        }
        else {
          innerLayers(layer).deltaHidden(innerLayers(layer + 1), delta(layer + 1), output)
        }
      }
      grads(layer) = innerLayers(layer).backprop(input, delta(layer))
    }
    val ce = crossEntropy(out.last, label)
    (grads, ce, batchSize.toDouble)
  }

  private def crossEntropy(out: BDM[Double], label: BDM[Double]): Double = {
    assert(label.rows == out.rows)
    assert(label.cols == out.cols)
    var cost = 0D
    for (i <- 0 until out.rows) {
      for (j <- 0 until out.cols) {
        val a = label(i, j)
        var b = out(i, j)
        if (b == 0) {
          b += 1e-15
        } else if (b == 1D) {
          b -= 1e-15
        }
        cost += a * math.log(b) + (1 - a) * math.log1p(1 - b)
      }
    }
    (0D - cost) / out.rows
  }

  private[mllib] def assign(newNN: NeuralNetwork): NeuralNetwork = {
    innerLayers.zip(newNN.innerLayers).foreach { case (oldLayer, newLayer) =>
      oldLayer.weight := newLayer.weight
      oldLayer.bias := newLayer.bias
    }
    this
  }

  def setSeed(seed: Long): Unit = {
    rand.setSeed(seed)
  }
}
