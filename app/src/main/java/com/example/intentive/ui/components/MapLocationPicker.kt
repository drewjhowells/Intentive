package com.example.intentive.ui.components

import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Check
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.example.intentive.data.model.LocationData
import com.google.android.gms.maps.model.CameraPosition
import com.google.android.gms.maps.model.LatLng
import com.google.maps.android.compose.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MapLocationPicker(
    initialLocation: LocationData?,
    onLocationSelected: (LocationData) -> Unit,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    
    // Default to user's location or a default location (you might want to get user's current location)
    val defaultLatLng = LatLng(40.7128, -74.0060) // New York City as default
    val initialLatLng = initialLocation?.toLatLng() ?: defaultLatLng
    
    var selectedLocation by remember { mutableStateOf(initialLatLng) }
    var cameraPositionState = rememberCameraPositionState {
        position = CameraPosition.fromLatLngZoom(initialLatLng, 15f)
    }
    
    Column(modifier = modifier.fillMaxSize()) {
        // Top app bar
        TopAppBar(
            title = { Text("Pick Location") },
            navigationIcon = {
                IconButton(onClick = onDismiss) {
                    Icon(Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                }
            },
            actions = {
                TextButton(
                    onClick = {
                        val locationData = LocationData(
                            address = "Selected Location: ${String.format("%.6f", selectedLocation.latitude)}, ${String.format("%.6f", selectedLocation.longitude)}",
                            latitude = selectedLocation.latitude,
                            longitude = selectedLocation.longitude
                        )
                        onLocationSelected(locationData)
                    }
                ) {
                    Icon(Icons.Default.Check, contentDescription = "Confirm")
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Confirm")
                }
            }
        )
        
        // Map
        Box(modifier = Modifier.weight(1f)) {
            GoogleMap(
                modifier = Modifier.fillMaxSize(),
                cameraPositionState = cameraPositionState,
                onMapClick = { latLng ->
                    selectedLocation = latLng
                }
            ) {
                // Marker for selected location
                Marker(
                    state = MarkerState(position = selectedLocation),
                    title = "Selected Location",
                    snippet = "Tap to confirm this location"
                )
            }
            
            // Instruction card
            Card(
                modifier = Modifier
                    .align(Alignment.TopCenter)
                    .padding(16.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
            ) {
                Text(
                    text = "Tap on the map to select a location",
                    modifier = Modifier.padding(12.dp),
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
        
        // Bottom info panel
        Card(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            elevation = CardDefaults.cardElevation(defaultElevation = 4.dp)
        ) {
            Column(
                modifier = Modifier.padding(16.dp)
            ) {
                Text(
                    text = "Selected Coordinates:",
                    style = MaterialTheme.typography.titleSmall
                )
                Text(
                    text = "Lat: ${String.format("%.6f", selectedLocation.latitude)}",
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(top = 4.dp)
                )
                Text(
                    text = "Lng: ${String.format("%.6f", selectedLocation.longitude)}",
                    style = MaterialTheme.typography.bodyMedium
                )
            }
        }
    }
}
