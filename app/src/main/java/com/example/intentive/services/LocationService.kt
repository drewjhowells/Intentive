package com.example.intentive.services

import android.content.Context
import android.util.Log
import com.example.intentive.data.model.LocationData
import com.example.intentive.data.model.LocationSuggestion
import com.google.android.gms.maps.model.LatLng
import com.google.android.libraries.places.api.Places
import com.google.android.libraries.places.api.model.AutocompletePrediction
import com.google.android.libraries.places.api.model.AutocompleteSessionToken
import com.google.android.libraries.places.api.model.Place
import com.google.android.libraries.places.api.net.FetchPlaceRequest
import com.google.android.libraries.places.api.net.FindAutocompletePredictionsRequest
import com.google.android.libraries.places.api.net.PlacesClient
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.suspendCancellableCoroutine
import javax.inject.Inject
import javax.inject.Singleton
import kotlin.coroutines.resume

@Singleton
class LocationService @Inject constructor(
    @ApplicationContext private val context: Context
) {
    
    companion object {
        private const val TAG = "LocationService"
    }
    
    private lateinit var placesClient: PlacesClient
    private var sessionToken = AutocompleteSessionToken.newInstance()
    
    fun initialize(apiKey: String) {
        try {
            if (!Places.isInitialized()) {
                Places.initialize(context, apiKey)
            }
            placesClient = Places.createClient(context)
            Log.d(TAG, "Places API initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing Places API", e)
        }
    }
    
    suspend fun searchLocations(query: String): List<LocationSuggestion> {
        if (!::placesClient.isInitialized) {
            Log.w(TAG, "Places client not initialized")
            return emptyList()
        }
        
        return suspendCancellableCoroutine { continuation ->
            val request = FindAutocompletePredictionsRequest.builder()
                .setSessionToken(sessionToken)
                .setQuery(query)
                .build()
            
            placesClient.findAutocompletePredictions(request)
                .addOnSuccessListener { response ->
                    val suggestions = response.autocompletePredictions.map { prediction ->
                        LocationSuggestion(
                            placeId = prediction.placeId,
                            description = prediction.getFullText(null).toString(),
                            mainText = prediction.getPrimaryText(null).toString(),
                            secondaryText = prediction.getSecondaryText(null).toString()
                        )
                    }
                    continuation.resume(suggestions)
                }
                .addOnFailureListener { exception ->
                    Log.e(TAG, "Error searching locations", exception)
                    continuation.resume(emptyList())
                }
        }
    }
    
    suspend fun getLocationDetails(placeId: String): LocationData? {
        if (!::placesClient.isInitialized) {
            Log.w(TAG, "Places client not initialized")
            return null
        }
        
        return suspendCancellableCoroutine { continuation ->
            val placeFields = listOf(
                Place.Field.ID,
                Place.Field.NAME,
                Place.Field.ADDRESS,
                Place.Field.LAT_LNG
            )
            
            val request = FetchPlaceRequest.newInstance(placeId, placeFields)
            
            placesClient.fetchPlace(request)
                .addOnSuccessListener { response ->
                    val place = response.place
                    val latLng = place.latLng
                    
                    if (latLng != null) {
                        val locationData = LocationData(
                            address = place.address ?: place.name ?: "Unknown Location",
                            latitude = latLng.latitude,
                            longitude = latLng.longitude,
                            placeId = place.id
                        )
                        continuation.resume(locationData)
                    } else {
                        Log.w(TAG, "No coordinates found for place: $placeId")
                        continuation.resume(null)
                    }
                }
                .addOnFailureListener { exception ->
                    Log.e(TAG, "Error fetching place details", exception)
                    continuation.resume(null)
                }
        }
    }
    
    fun renewSessionToken() {
        sessionToken = AutocompleteSessionToken.newInstance()
    }
    
    suspend fun reverseGeocode(latLng: LatLng): LocationData? {
        // For reverse geocoding, you would typically use the Geocoder API
        // This is a simplified implementation
        return LocationData(
            address = "Location: ${String.format("%.6f", latLng.latitude)}, ${String.format("%.6f", latLng.longitude)}",
            latitude = latLng.latitude,
            longitude = latLng.longitude
        )
    }
}
