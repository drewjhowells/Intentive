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
        // Initialize location service with API key from BuildConfig
        locationService.initialize(BuildConfig.API_KEY)
    }
}