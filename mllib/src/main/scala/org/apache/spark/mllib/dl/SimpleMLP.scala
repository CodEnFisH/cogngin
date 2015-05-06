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

import breeze.linalg.{DenseVector => BDV, DenseMatrix => BDM, axpy => brzAxpy,
argmax => brzArgMax, norm => brzNorm}
import org.apache.spark.mllib.linalg.{Vector => SV, DenseVector => SDV, Vectors, BLAS}
import org.apache.spark.mllib.optimization.{Updater, Gradient, GradientDescent}
import org.apache.spark.rdd.RDD

object SimpleMLP {

  def train(
             data: RDD[(SV, SV)],
             batchSize: Int,
             numIteration: Int,
             topology: Array[Int],
             fraction: Double,
             weightCost: Double,
             learningRate: Double
             ): NeuralNetwork = {
    runSGD(data, new NeuralNetwork(SimpleMLP.initialize(topology)),
      batchSize, numIteration, fraction, weightCost, learningRate)
  }

  def runSGD(
              data: RDD[(SV, SV)],
              nn: NeuralNetwork,
              batchSize: Int,
              maxNumIterations: Int,
              fraction: Double,
              regParam: Double,
              learningRate: Double): NeuralNetwork = {
    val gradient = new SimpleMLPGradient(nn)
    val updater = new SimpleMLPAdaDeltaUpdater(nn.topology)
    val optimizer = new GradientDescent(gradient, updater).
      setMiniBatchFraction(fraction).
      setNumIterations(maxNumIterations).
      setRegParam(regParam).
      setStepSize(learningRate)

    val trainingRDD = toTrainingRDD(data, batchSize, nn.numInput, nn.numOut)
    val weights = optimizer.optimize(trainingRDD, transformModelToVector(nn))
    rebuildModelFromVector(nn, weights)
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

  private[mllib] def rebuildModelFromVector(mlp: NeuralNetwork, weights: SV): Unit = {
    val structure = vectorToStructure(mlp.topology, weights)
    val layers: Array[Layer] = mlp.innerLayers
    for (i <- 0 until structure.length) {
      val (weight, bias) = structure(i)
      val layer = layers(i)
      layer.setWeight(weight)
      layer.setBias(bias)
    }
  }

  private[mllib] def transformModelToVector(nn: NeuralNetwork): SV = {
    foldToVector(nn.innerLayers.map(l => (l.weight, l.bias)))
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

private[mllib] class SimpleMLPGradient(
                                        val nn: NeuralNetwork,
                                        batchSize: Int = 1) extends Gradient {

  val numIn = nn.numInput
  val numLabel = nn.numOut

  override def compute(data: SV, label: Double, weights: SV): (SV, Double) = {

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

    SimpleMLP.rebuildModelFromVector(nn, weights)
    // for ps: the grads might not be necessary.
    // get the grads from cassandra.
    var (grads, error, numCol) = nn.iterate(input, label)
    // put the get grads function here. Like the foldToVector thing.
    if (numCol != 1D) {
      val scale = 1D / numCol
      grads.foreach { t =>
        t._1 :*= scale
        t._2 :*= scale
      }
      error *= scale
    }

    (SimpleMLP.foldToVector(grads), error)
  }

  override def compute(
                        data: SV,
                        label: Double,
                        weights: SV,
                        cumGradient: SV): Double = {
    val (grad, err) = compute(data, label, weights)
    cumGradient.toBreeze += grad.toBreeze
    err
  }
}

private[mllib] class SimpleMLPUpdater(val topology: Array[Int]) extends Updater {

  protected val numLayer = topology.length - 1

  protected val sumWeightsLen: Int = {
    var sumLen = 0
    for (layer <- 0 until numLayer) {
      val numRow = topology(layer + 1)
      val numCol = topology(layer)
      sumLen += numRow * numCol + numRow
    }
    sumLen
  }

  protected def l2(
                    weightsOld: SV,
                    gradient: SV,
                    stepSize: Double,
                    iter: Int,
                    regParam: Double): Double = {
    if (regParam > 0D) {
      val nn = SimpleMLP.vectorToStructure(topology, weightsOld)
      val grads = SimpleMLP.vectorToStructure(topology, gradient)
      for (layer <- 0 until nn.length) {
        brzAxpy(regParam, nn(layer)._1, grads(layer)._1)
      }
    }
    regParam
  }

  override def compute(
                        weightsOld: SV,
                        gradient: SV,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (SV, Double) = {
    l2(weightsOld, gradient, stepSize, iter, regParam)
    BLAS.axpy(-stepSize, gradient, weightsOld)

    val norm = brzNorm(weightsOld.toBreeze, 2.0)
    (weightsOld, 0.5 * regParam * norm * norm)
  }
}

private[mllib] class SimpleMLPAdaDeltaUpdater(
  topology: Array[Int],
  rho: Double = 0.99,
  epsilon: Double = 1e-8) extends SimpleMLPUpdater(topology) {

  assert(rho > 0 && rho < 1)
  lazy val gradientSum = {
    new SDV(new Array[Double](sumWeightsLen))
  }

  lazy val deltaSum = {
    new SDV(new Array[Double](sumWeightsLen))
  }

  override def compute(
                        weightsOld: SV,
                        gradient: SV,
                        stepSize: Double,
                        iter: Int,
                        regParam: Double): (SV, Double) = {
    l2(weightsOld, gradient, stepSize, iter, regParam)
    val grad = gradient.toBreeze

    val g2 = grad :* grad
    this.synchronized {
      BLAS.scal(rho, gradientSum)
      BLAS.axpy(1 - rho, Vectors.fromBreeze(g2), gradientSum)
    }

    for (i <- 0 until grad.length) {
      val rmsDelta = math.sqrt(epsilon + deltaSum(i))
      val rmsGrad = math.sqrt(epsilon + gradientSum(i))
      grad(i) *= (rmsDelta / rmsGrad)
    }

    val d2 = grad :* grad
    this.synchronized {
      BLAS.scal(rho, deltaSum)
      BLAS.axpy(1 - rho, Vectors.fromBreeze(d2), deltaSum)
    }

    BLAS.axpy(-stepSize, Vectors.fromBreeze(grad), weightsOld)
    (weightsOld, regParam)
  }

}
