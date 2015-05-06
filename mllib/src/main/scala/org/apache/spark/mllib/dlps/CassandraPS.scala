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

import breeze.linalg.{DenseMatrix, DenseVector}
import com.datastax.spark.connector._
import com.datastax.spark.connector.cql.{CassandraConnector, TableDef => CTableDef}
import org.apache.spark.util.Utils
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by maxmin on 3/16/15.
 * To serve as parameter server interface
 */

object CassandraPS {

  def apply(conf: SparkConf,
            layerNum: Int,
            neuronNums: Array[Int]):
  (SparkContext, CassandraPS) = {
    val hosts = conf.get("cogngin.cassandra.host", "10.227.119.245")
    val user = conf.get("cogngin.cassandra.user", "psUser")
    val passwd = conf.get("cogngin.cassandra.password", "parameter")

    conf.set("spark.cassandra.connection.host", hosts)
      .set("spark.cassandra.auth.username", user)
      .set("spark.cassandra.auth.password", passwd)

    val sc = new SparkContext(conf)
    val psMeta = PSMeta(Array(hosts), generateKeySpace(), generateTableName())
    val modelMeta = ModelMeta(layerNum, neuronNums)
    val cred = PSCredential(user, passwd)
    val ps = new CassandraPS(sc, psMeta, modelMeta, cred)
    ps.initPS(conf)
    (sc, ps)
  }

  def generateKeySpace() = {
    "test"
  }

  def generateTableName() = {
    "modeltest"
  }
}

/**
 * This CassandraPS is still only interfacing with cassandra.
 * This class should be generalized into a Parameter Server Interface,
 * support initPS, initWeight, consistency model
 * and some other featured APIs for PS.
 */
class CassandraPS(sc: SparkContext, psMeta: PSMeta,
                  modelMeta: ModelMeta, cred: PSCredential) {

  def keyspaceName() = psMeta.keyspaceName

  def tableName() = psMeta.tableName

  def getPSMeta = psMeta

  def getPSModelMeta = modelMeta

  def getCred = cred

  def initPS(conf: SparkConf) = {
    // todo: create tables, etc.
    val createCmd = "CREATE TABLE %s.%s ".format(psMeta.keyspaceName, psMeta.tableName) +
                    "(layerid int, rowid int, colid int, " +
                    "bias double, biasG list<double>, " +
                    "value double, weightG list<double>, " +
                    "PRIMARY KEY (layerid, rowid, colid));"
    CassandraConnector(conf).withSessionDo { session =>
      session.execute(createCmd)
    }
  }

  def randomInit() = {
    // todo: random initialize weights and bias.
    var inserts: List[(Int, Int, Int, Double, Double)] = Nil

    for(layer <- 1 until modelMeta.layerNum) {
      val in = modelMeta.neuronNum(layer-1)
      val out = modelMeta.neuronNum(layer)
      val v = if (layer < modelMeta.layerNum-1) 4D * math.sqrt(6D / (in + out))
              else 0D
      for(row <- 1 to out) {
        for(col <- 1 to in) {
          // create in mem collection and use spark cassandra to insert
          val randomW = (2D * Utils.random.nextDouble() - 1D) * v
          inserts = (layer, row, col, 0D, randomW) :: inserts
        }
      }
    }
    sc.parallelize(inserts).saveToCassandra(psMeta.keyspaceName, psMeta.tableName,
    SomeColumns("layerid", "rowid", "colid", "bias", "value"))
  }

  def randomInit(low: Double, up: Double, bias: Double) = {
    // todo: random initialize weights and bias.
    var inserts: List[(Int, Int, Int, Double, Double)] = Nil

    for(layer <- 1 until modelMeta.layerNum) {
      val in = modelMeta.neuronNum(layer-1)
      val out = modelMeta.neuronNum(layer)
      for(row <- 1 to out) {
        for(col <- 1 to in) {
          // create in mem collection and use spark cassandra to insert
          val randomW = (up-low) * Utils.random.nextDouble() + low
          inserts = (layer, row, col, bias, randomW) :: inserts
        }
      }
    }
    sc.parallelize(inserts).saveToCassandra(psMeta.keyspaceName, psMeta.tableName,
      SomeColumns("layerid", "rowid", "colid", "bias", "value"))
  }

  def initWithValues(paras: Array[(DenseMatrix[Double], DenseVector[Double])]) = {
    // todo: use the give data to initialize the weights and bias.
    assert(paras.length == modelMeta.layerNum-1)
    var inserts: List[(Int, Int, Int, Double, Double)] = Nil

    for(layer <- 1 until modelMeta.layerNum) {
      val in = modelMeta.neuronNum(layer-1)
      val out = modelMeta.neuronNum(layer)
      val wM = paras(layer-1)._1
      val bV = paras(layer-1)._2
      for(row <- 1 to out) {
        for(col <- 1 to in) {
          // create in mem collection and use spark cassandra to insert
          inserts = (layer, row, col, bV(row-1), wM(row-1, col-1)) :: inserts
        }
      }
    }
    sc.parallelize(inserts).saveToCassandra(psMeta.keyspaceName, psMeta.tableName,
      SomeColumns("layerid", "rowid", "colid", "bias", "value"))
  }

  def mergeAndUpdate(rate: Double) = {
    val rows = sc.cassandraTable(psMeta.keyspaceName, psMeta.tableName)

    rows.map(r => {
      // map these old rows to new ones and call saveToCassandra func.
      // test if the saveToCassandra works in the test suite.
      val gredList = r.get[List[Double]]("weightg")
      val gredAgg = gredList.fold(0.0) ((s, d) => s + d) / gredList.size
      val newValue = r.get[Double]("value") + rate * gredAgg


      val newBias =
        if(r.get[Int]("colid") == 1) {
          val list = r.get[List[Double]]("biasg")
          val biasAgg = list.fold(0.0)((b, db) => b + db) / list.size
          r.get[Double]("bias") + rate * biasAgg
        }
        else {
          0.0
        }

      ( r.get[Int]("layerid"),
        r.get[Int]("rowid"),
        r.get[Int]("colid"),
        newBias,
        null,
        newValue,
        null)
    }).saveToCassandra(psMeta.keyspaceName, psMeta.tableName,
        SomeColumns("layerid", "rowid", "colid", "bias", "biasg", "value", "weightg"))
  }
}
