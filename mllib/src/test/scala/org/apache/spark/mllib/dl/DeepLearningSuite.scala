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

import scala.util.Random
import org.apache.spark.mllib.util.MnistDatasetSuite
import org.scalatest.FunSuite

class DeepLearningSuite extends FunSuite with MnistDatasetSuite {

//  test("MnistDatasetReader test") {
//    val (data, numVisible) = mnistTrainDataset(10, 0)
//    println(s"the data size is ${data.count()}")
//    println(s"the numVisible is $numVisible")
//  }

  def runThreeLayerSimpleMLPTest(numIter: Int, numSample: Int): Unit = {
    println("====Starting Test====")
    println(s"Number of iterations: $numIter")
    println(s"Number of samples: $numSample")
    val (data, numVisible) = mnistTrainDataset(numSample)
    data.cache()
    println("input data is cached!")
    val nn = SimpleMLP.train(data, 20, numIter, Array(numVisible, 500, 100, 10), 0.05, 0.0, 0.01)
    //SimpleMLP.runSGD(data, nn, 37, 6000, 0.1, 0.0, 0.01)

    val (dataTest, _) = mnistTrainDataset(10000, 5000)
    println("Error: " + SimpleMLP.error(dataTest, nn, 100))
    println("====Done Test====/n")
  }

  test("DL test") {
    val sampleNumbers = Array(2500, 5000, 10000)
    val iterNumbers = Array(100, 1000, 2000, 3000)
    for(i <- 0 until sampleNumbers.length) {
      for(j <- 0 until iterNumbers.length) {
        runThreeLayerSimpleMLPTest(iterNumbers(j), sampleNumbers(i))
      }
    }
  }
}

