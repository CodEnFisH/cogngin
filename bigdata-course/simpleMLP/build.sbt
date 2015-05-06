name := "simpleMLP"

version := "1.0"

scalaVersion := "2.10.4"

resolvers += Resolver.mavenLocal

//externalResolvers :=
//  Resolver.withDefaultResolvers(resolvers.value, mavenCentral = false)

libraryDependencies += "org.apache.spark" %% "spark-core" % "1.1.0"

libraryDependencies += "org.apache.spark" %% "spark-mllib" % "1.1.0"
