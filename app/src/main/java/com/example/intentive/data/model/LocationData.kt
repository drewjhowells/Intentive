package com.example.intentive.data.model

import com.google.android.gms.maps.model.LatLng

data class LocationData(
    val address: String,
    val latitude: Double,
    val longitude: Double,
    val placeId: String? = null
) {
    fun toLatLng(): LatLng = LatLng(latitude, longitude)
}

data class LocationSuggestion(
    val placeId: String,
    val description: String,
    val mainText: String,
    val secondaryText: String
)
