package com.example.intentive

import android.app.Application
import com.example.intentive.services.LocationService
import dagger.hilt.android.HiltAndroidApp
import javax.inject.Inject

@HiltAndroidApp
class IntentiveApp : Application() {
    
    @Inject
    lateinit var locationService: LocationService
    
    override fun onCreate() {
        super.onCreate()
        // Initialize location service with API key
        // Note: Replace with your actual Google Maps API key
        locationService.initialize("AIzaSyA_UmNb7WRVG87At3OotBxomvQJ8muPQd0")
    }
}