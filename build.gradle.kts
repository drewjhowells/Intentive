// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
    alias(libs.plugins.android.application) apply false // Changed
    alias(libs.plugins.kotlin.android) apply false    // Changed
    alias(libs.plugins.kotlin.kapt) apply false       // Changed
    alias(libs.plugins.hilt.gradle) apply false       // Changed
    // alias(libs.plugins.kotlin.compose.compiler) apply false // If using Kotlin 2.0+ and new Compose Compiler plugin
    // alias(libs.plugins.ksp) apply false // If using KSP
}