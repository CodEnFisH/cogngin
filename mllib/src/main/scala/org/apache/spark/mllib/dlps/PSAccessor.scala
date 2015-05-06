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

import breeze.linalg.{DenseVector, DenseMatrix}
import com.datastax.driver.core._
import com.datastax.driver.core.querybuilder.QueryBuilder
import com.datastax.driver.core.querybuilder.QueryBuilder.{eq => ceq}
import scala.collection.JavaConversions._

/**
 * Created by maxmin on 3/17/15.
 * PSAccessor
 * Task side accessor for the cassandra
 */

case class PSMeta(psPoints: Array[String],
                  keyspaceName: String,
                  tableName: String)
case class ModelMeta(layerNum: Int,
                     neuronNum: Array[Int])
case class PSCredential(user: String, passwd: String)

class PSAccessor(modelMeta: ModelMeta, psMeta: PSMeta, cred: PSCredential) {

  lazy val (psCluster: Cluster, session: Session) = getPSCluster
  val BATCH_SIZE = 20 // controls the batch update size.

  private def getPSCluster: (Cluster, Session) = {
    val cluster = Cluster.builder()
      .addContactPoints(psMeta.psPoints.mkString(","))
      .withCredentials(cred.user, cred.passwd)
      .build()
    (cluster, cluster.connect(psMeta.keyspaceName))
  }

  def getLayer(layerID: Int): DenseMatrix[Double] = {
    assert(layerID > 0 && layerID < modelMeta.layerNum)
    val m = DenseMatrix.zeros[Double](modelMeta.neuronNum(layerID),
      modelMeta.neuronNum(layerID-1))
    // fetch the corresponding values to m.
    // use get row and assign to corresponding values.
    val query = QueryBuilder.select("rowid", "colid", "value")
      .from(psMeta.tableName)
      .where(ceq("layerid", layerID))
    session.execute(query).all().foreach(r => {
      m(r.getInt("rowid")-1, r.getInt("colid")-1) = r.getDouble("value")
    })
    m
  }

  def getRowOfLayer(layerID: Int, rowID: Int): DenseVector[Double] = {
    assert(layerID > 0 && layerID < modelMeta.layerNum)
    assert(rowID >= 0 && rowID < modelMeta.neuronNum(layerID))
    val v = DenseVector.zeros[Double](modelMeta.neuronNum(layerID-1))

    // fetch values and assign to v
    val query = QueryBuilder.select("colid", "value")
      .from(psMeta.tableName)
      .where(ceq("layerid", layerID))
      .and(ceq("rowid", rowID))
    session.execute(query).all().foreach(r => {
      v(r.getInt("colid")-1) = r.getDouble("value")
    })
    v
  }

  def getOne(layerID: Int, rowID: Int, colID: Int): Double = {
    assert(layerID > 0 && layerID <= modelMeta.layerNum)
    assert(rowID > 0 && rowID <= modelMeta.neuronNum(rowID))
    assert(colID > 0 && colID <= modelMeta.neuronNum(rowID-1))
    val query = QueryBuilder.select("value")
      .from(psMeta.tableName)
      .where(ceq("layerid", layerID))
      .and(ceq("rowid", rowID))
      .and(ceq("colid", colID))
    session.execute(query).one().getDouble(0)
  }

  def getBiasOfLayer(layerID: Int): DenseVector[Double] = {
    assert(layerID > 0 && layerID < modelMeta.layerNum)
    val v = DenseVector.zeros[Double](modelMeta.neuronNum(layerID))

    // fetch values and assign to v
    val query = QueryBuilder.select("rowid", "colid", "bias")
      .from(psMeta.tableName)
      .where(ceq("layerid", layerID))
      //.and(ceq("colid", 1))
    session.execute(query).all().foreach(r => {
      if(r.getInt("colid") == 1) {
        v(r.getInt("rowid") - 1) = r.getDouble("bias")
      }
    })
    v
  }

  // use batch
  // setLayerDeltas can be merged with setRowDeltasOfLayer
  def setLayerWeightUpdate(layerID: Int, wGMat: DenseMatrix[Double]) = {
    var batch = QueryBuilder.batch()
    var count = 0
    // for loop and batch add
    for(i <- 1 to modelMeta.neuronNum(layerID)) {
      for(j <- 1 to modelMeta.neuronNum(layerID-1)) {
        batch.add(generateWeightUpdate(layerID, i, j, wGMat(i-1, j-1)))
        count += 1
        if (count >= BATCH_SIZE) {
          count = 0
          session.execute(batch)
          batch = QueryBuilder.batch()
        }
      }
    }
    session.execute(batch)
  }

  def setRowWeightUpdateOfLayer(layerID: Int, rowID: Int, wGVec: DenseVector[Double]) = {
    var batch = QueryBuilder.batch()
    var count = 0
    // for loop and batch add
    for(j <- 1 to modelMeta.neuronNum(layerID-1)) {
      batch.add(generateWeightUpdate(layerID, rowID, j, wGVec(j-1)))
      count += 1
      if (count >= BATCH_SIZE) {
        count = 0
        session.execute(batch)
        batch = QueryBuilder.batch()
      }
    }
    session.execute(batch)
  }

  def setOneWeightUpdate(layerID: Int, rowID: Int, colID: Int, wG: Double) = {
    assert(layerID > 0 && layerID <= modelMeta.layerNum)
    assert(rowID > 0 && rowID <= modelMeta.neuronNum(rowID))
    assert(colID > 0 && colID <= modelMeta.neuronNum(rowID-1))

    session.execute(generateWeightUpdate(layerID, rowID, colID, wG))
  }

  def setBiasOfLayer(layerID: Int, bGVec: DenseVector[Double]) = {
    var batch = QueryBuilder.batch()
    var count = 0
    // for loop and batch add
    for(row <- 1 to modelMeta.neuronNum(layerID)) {
      batch.add(generateBiasUpdate(layerID, row, 1, bGVec(row-1)))
      count += 1
      if (count >= BATCH_SIZE) {
        count = 0
        session.execute(batch)
        batch = QueryBuilder.batch()
      }
    }
    session.execute(batch)
  }

  def setOneBiasUpdate(layerID: Int, rowID: Int, bG: Double) = {
    assert(layerID > 0 && layerID <= modelMeta.layerNum)
    assert(rowID > 0 && rowID <= modelMeta.neuronNum(rowID))

    session.execute(generateWeightUpdate(layerID, rowID, 1, bG))
  }

  def generateWeightUpdate(layerID: Int, rowID: Int,
                           colID: Int, weightG: Double) =
    generateAppendStatement(false, layerID, rowID, colID, weightG)

  def generateBiasUpdate(layerID: Int, rowID: Int,
                         colID: Int, biasG: Double) =
    generateAppendStatement(true, layerID, rowID, colID, biasG)

  def generateAppendStatement(bias: Boolean, layerID: Int, rowID: Int,
                              colID: Int, delta: Double) = {
    val colName = if(bias) "biasg" else "weightg"
    val query = QueryBuilder.update(psMeta.tableName)
      .`with`(QueryBuilder.append(colName, delta))
      .where(ceq("layerid", layerID))
      .and(ceq("rowid", rowID))
      .and(ceq("colid", colID))
    query
  }

  def close() = {
    psCluster.close()
  }
}
