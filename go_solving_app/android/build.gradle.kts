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

// Workaround: pub.dev `onnxruntime` plugin (Android library, transitive dep
// via katago_onnx_mobile) hardcodes compileSdk=33 in its own build.gradle,
// but other transitive AndroidX libs resolved into the build (androidx.core
// 1.13.1, androidx.fragment 1.7.1, etc.) require compileSdk>=34. Raise the
// onnxruntime module's compileSdk to match this app's compileSdk so its own
// `checkReleaseAarMetadata` task passes.
subprojects {
    if (project.name == "onnxruntime") {
        plugins.withId("com.android.library") {
            extensions.configure<com.android.build.gradle.LibraryExtension> {
                compileSdk = 36
            }
        }
    }
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
