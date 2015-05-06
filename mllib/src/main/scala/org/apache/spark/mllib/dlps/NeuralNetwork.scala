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
class NeuralNetwork(innerLayers: Array[Layer],
                    psMeta: PSMeta,
                    psCred: PSCredential) extends Logging with Serializable {

  val topology: Array[Int] = {
    val topo = new Array[Int](numLayer + 1)
    topo(0) = numInput
    for (i <- 1 to numLayer) {
      topo(i) = innerLayers(i - 1).outNum
    }
    topo
  }

  // each task should use their own ps accessor. This way of init is because of a scala bug
  @transient
  lazy val psAccessor: PSAccessor = initPSAccessor

  private def initPSAccessor =  new PSAccessor(ModelMeta(topology.length, topology), psMeta, psCred)

  def numLayer = innerLayers.length

  def numInput = innerLayers.head.inNum

  def numOut = innerLayers.last.outNum

  def predict(x: BDenseMatrix[Double]): BDenseMatrix[Double] = {
    var output = x
    for (layer <- 0 until numLayer) {
      output = innerLayers(layer).forward(
        output,
        psAccessor.getLayer(layer + 1),
        psAccessor.getBiasOfLayer(layer + 1))
    }
    output
  }

  /**
   *
   * @param x, mini-batch of sample vectors, each column is a vector
   * @param label, the true label
   * @return grads
   */
  private[mllib] def iterate(
    x: BDenseMatrix[Double],
    label: BDenseMatrix[Double]):
  Double = {
    val batchSize = x.cols
    val in = new Array[BDenseMatrix[Double]](numLayer)
    val out = new Array[BDenseMatrix[Double]](numLayer)
    val weight = new Array[BDenseMatrix[Double]](numLayer)
    val bias = new Array[BDenseVector[Double]](numLayer)
    val delta = new Array[BDenseMatrix[Double]](numLayer)

    // Min's note
    // this computation is using the same model for the whole batch of data
    for (layer <- 0 until numLayer) {
      println("#### Task side, Layer %d".format(layer + 1))
      val input = if (layer == 0) {
        x
      } else {
        out(layer - 1)
      }
      in(layer) = input

      var timer = System.currentTimeMillis()
      weight(layer) = psAccessor.getLayer(layer + 1)
      bias(layer) = psAccessor.getBiasOfLayer(layer + 1)
      println("#### Task side, Layer %d, time for fetch: %s"
        .format(layer + 1, System.currentTimeMillis() - timer))

      timer = System.currentTimeMillis()
      val output = innerLayers(layer).forward(
        input,
        weight(layer),
        bias(layer))
      out(layer) = output
      println("#### Task side, Layer %d, time for compute: %s"
        .format(layer + 1, System.currentTimeMillis() - timer))
    }

    for (layer <- (0 until numLayer).reverse) {
      var timer = System.currentTimeMillis()
      val input = in(layer)
      val output = out(layer)
      delta(layer) = {
        if (layer == numLayer - 1) {
          innerLayers(layer).deltaOutput(output, label)
        }
        else {
          innerLayers(layer).deltaHidden(weight(layer + 1), delta(layer + 1), output)
        }
      }
      val grads = innerLayers(layer).backprop(input, delta(layer))
      if (batchSize != 1D) {
        val scale = 1D / batchSize
        grads._1 :*= scale
        grads._2 :*= scale
      }
      println("#### Task side, Layer %d, time for backprop: %s"
        .format(layer + 1, System.currentTimeMillis() - timer))
      timer = System.currentTimeMillis()
      psAccessor.setLayerWeightUpdate(layer + 1, grads._1)
      psAccessor.setBiasOfLayer(layer + 1, grads._2)
      println("#### Task side, Layer %d, time for update: %s"
        .format(layer + 1, System.currentTimeMillis() - timer))
    }
    val ce = crossEntropy(out.last, label)
    ce / batchSize
  }

  private def crossEntropy(out: BDenseMatrix[Double], label: BDenseMatrix[Double]): Double = {
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

  // this is no longer needed if ps is used
  /*
  private[mllib] def assign(newNN: NeuralNetwork): NeuralNetwork = {
    innerLayers.zip(newNN.innerLayers).foreach { case (oldLayer, newLayer) =>
      oldLayer.weight := newLayer.weight
      oldLayer.bias := newLayer.bias
    }
    this
  }*/
}
