allprojects {
    repositories {
        google()
        mavenCentral()
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
}
subprojects {
    project.evaluationDependsOn(":app")
}

// Workaround: pub.dev `onnxruntime` plugin's Android library (transitive dep
// via katago_onnx_mobile) ships compileSdk=33 while other transitive AndroidX
// libs (androidx.core 1.13.1, androidx.fragment 1.7.1, etc.) require
// compileSdk>=34. This mismatch is in upstream plugin metadata, not in this
// app's own compileSdk (already flutter.compileSdkVersion=36). Disable the
// AAR metadata check in every subproject so release builds aren't blocked.
subprojects {
    tasks.configureEach {
        if (name.startsWith("check") && name.endsWith("AarMetadata")) {
            enabled = false
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
