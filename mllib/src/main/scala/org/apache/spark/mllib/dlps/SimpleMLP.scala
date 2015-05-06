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

import breeze.linalg.{DenseMatrix => BDM, DenseVector => BDV, argmax => brzArgMax, axpy => brzAxpy, norm => brzNorm}
import org.apache.spark.mllib.linalg.{DenseVector => SDV, Vector => SV}
import org.apache.spark.rdd.RDD

object SimpleMLP {

  def train(
             psHandler: CassandraPS,
             data: RDD[(SV, SV)],
             batchSize: Int,
             numIteration: Int,
             fraction: Double,
             weightCost: Double,
             learningRate: Double
             ): NeuralNetwork = {
    psHandler.randomInit()
    runSGD(
      psHandler,
      data,
      new NeuralNetwork(SimpleMLP.initialize(psHandler.getPSModelMeta.neuronNum),
        psHandler.getPSMeta,
        psHandler.getCred),
      batchSize,
      numIteration,
      fraction,
      weightCost,
      learningRate)
  }

  def runSGD(
              psHandler: CassandraPS,
              data: RDD[(SV, SV)],
              nn: NeuralNetwork,
              batchSize: Int,
              maxNumIterations: Int,
              fraction: Double,
              regParam: Double,
              learningRate: Double): NeuralNetwork = {
    val gradient = new PSSimpleGradient(nn)
    val optimizer = new PSGradientDescent(psHandler, gradient).
      setMiniBatchFraction(fraction).
      setNumIterations(maxNumIterations).
      setRegParam(regParam).
      setStepSize(learningRate)

    val trainingRDD = toTrainingRDD(data, batchSize, nn.numInput, nn.numOut)
    optimizer.optimize(trainingRDD)
    nn
  }

  def initialize(topology: Array[Int]): Array[Layer] = {
    val numLayer = topology.length - 1
    val layers = new Array[Layer](numLayer)
    var nextLayer: Layer = null
    for (layer <- (0 until numLayer).reverse) {
      layers(layer) = if (layer == numLayer - 1) {
        new SoftmaxLayer(topology(layer), topology(layer + 1))
      }
      else {
        new SigmoidLayer(topology(layer), topology(layer + 1))
      }
      nextLayer = layers(layer)
      println(s"layers($layer) = ${layers(layer).inNum} * ${layers(layer).outNum}")
    }
    layers
  }

  private def toTrainingRDD(
                             data: RDD[(SV, SV)],
                             batchSize: Int,
                             numInput: Int,
                             numOut: Int): RDD[(Double, SV)] = {
    if (batchSize > 1) {
      batchVector(data, batchSize, numInput, numOut).map(t => (0D, t))
    }
    else {
      data.map { case (input, label) =>
        val sumLen = input.size + label.size
        val data = new Array[Double](sumLen)
        var offset = 0
        System.arraycopy(input.toArray, 0, data, offset, input.size)
        offset += input.size

        System.arraycopy(label.toArray, 0, data, offset, label.size)
        offset += label.size

        (0D, new SDV(data))
      }
    }
  }

  private[mllib] def foldToVector(grads: Array[(BDM[Double], BDV[Double])]): SV = {
    val numLayer = grads.length
    val sumLen = grads.map(m => m._1.rows * m._1.cols + m._2.length).sum
    val data = new Array[Double](sumLen)
    var offset = 0
    for (l <- 0 until numLayer) {
      val (gradWeight, gradBias) = grads(l)
      val numIn = gradWeight.cols
      val numOut = gradWeight.rows
      System.arraycopy(gradWeight.toArray, 0, data, offset, numOut * numIn)
      offset += numIn * numOut
      System.arraycopy(gradBias.toArray, 0, data, offset, numOut)
      offset += numOut
    }
    new SDV(data)
  }

  private[mllib] def vectorToStructure(
                                        topology: Array[Int],
                                        weights: SV): Array[(BDM[Double], BDV[Double])] = {
    val data = weights.toArray
    val numLayer = topology.length - 1
    val grads = new Array[(BDM[Double], BDV[Double])](numLayer)
    var offset = 0
    for (layer <- 0 until numLayer) {
      val numIn = topology(layer)
      val numOut = topology(layer + 1)
      val weight = new BDM[Double](numOut, numIn, data, offset)
      offset += numIn * numOut
      val bias = new BDV[Double](data, offset, 1, numOut)
      offset += numOut
      grads(layer) = (weight, bias)
    }
    grads
  }

  private[mllib] def batchVector(
                                  data: RDD[(SV, SV)],
                                  batchSize: Int,
                                  numInput: Int,
                                  numOut: Int): RDD[SV] = {
    batchMatrix(data, batchSize, numInput, numOut).map { t =>
      val input = t._1
      val label = t._2
      val sumLen = (input.rows + label.rows) * input.cols
      val data = new Array[Double](sumLen)
      var offset = 0
      System.arraycopy(input.toArray, 0, data, offset, input.rows * input.cols)
      offset += input.rows * input.cols

      System.arraycopy(label.toArray, 0, data, offset, label.rows * input.cols)
      offset += label.rows * label.cols
      new SDV(data)
    }
  }

  private[mllib] def batchMatrix(
                                  data: RDD[(SV, SV)],
                                  batchSize: Int,
                                  numInput: Int,
                                  numOut: Int): RDD[(BDM[Double], BDM[Double])] = {
    val dataBatch = data.mapPartitions { itr =>
      itr.grouped(batchSize).map { seq =>
        val x = BDM.zeros[Double](numInput, seq.size)
        val y = BDM.zeros[Double](numOut, seq.size)
        seq.zipWithIndex.foreach { case (v, i) =>
          x(::, i) :+= v._1.toBreeze
          y(::, i) :+= v._2.toBreeze
        }
        (x, y)
      }
    }
    dataBatch
  }

  def error(data: RDD[(SV, SV)], nn: NeuralNetwork, batchSize: Int): Double = {
    val count = data.count()
    val dataBatches = batchMatrix(data, batchSize, nn.numInput, nn.numOut)
    val sumError = dataBatches.map{case (x, y) =>
      val h = nn.predict(x)
      (0 until h.cols).map(i => {
        if (brzArgMax(y(::, i)) == brzArgMax(h(::, i))) 0D else 1D
      }).sum
    }.collect().sum
    sumError / count
  }
}

