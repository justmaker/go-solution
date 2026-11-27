plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.gosolving.go_solving_app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = flutter.ndkVersion

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    defaultConfig {
        // TODO: Specify your own unique Application ID (https://developer.android.com/studio/build/application-id.html).
        applicationId = "com.gosolving.go_solving_app"
        // You can update the following values to match your application needs.
        // For more information, see: https://flutter.dev/to/review-gradle-config.
        minSdk = flutter.minSdkVersion
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName
    }

    packaging {
        jniLibs {
            pickFirsts.add("lib/**/libonnxruntime.so")
        }
    }

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
}

flutter {
    source = "../.."
}

// Workaround: pub.dev `onnxruntime` plugin's Android library (katago_onnx_mobile
// transitive dep) ships with compileSdk=33 while transitive AndroidX libs
// (androidx.core 1.13.1, androidx.fragment 1.7.1, etc.) require compileSdk>=34.
// This conflict is in the upstream onnxruntime package metadata, not in this
// app's own compileSdk (already flutter.compileSdkVersion=36). Disable the
// AAR metadata check so release builds aren't blocked by this third-party
// library mismatch.
tasks.configureEach {
    if (name.startsWith("check") && name.endsWith("AarMetadata")) {
        enabled = false
    }
}
