package com.example.intentive.ui.components

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.MyLocation
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.intentive.data.model.LocationData
import com.example.intentive.data.model.LocationSuggestion
import kotlinx.coroutines.delay

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun LocationSearchField(
    location: String,
    onLocationChange: (String) -> Unit,
    onLocationSelected: (LocationData) -> Unit,
    onSuggestionSelected: (LocationSuggestion) -> Unit = {},
    onShowMapPicker: () -> Unit,
    suggestions: List<LocationSuggestion>,
    isLoading: Boolean = false,
    notifyAtLocation: Boolean = false,
    onLocationNotificationToggle: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    var showSuggestions by remember { mutableStateOf(false) }
    val keyboardController = LocalSoftwareKeyboardController.current

    Column(modifier = modifier) {
        // Search field with buttons
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedTextField(
                value = location,
                onValueChange = { 
                    onLocationChange(it)
                    showSuggestions = it.isNotBlank()
                },
                label = { Text("Location") },
                placeholder = { Text("Search for a location...") },
                leadingIcon = {
                    Icon(
                        Icons.Default.LocationOn,
                        contentDescription = "Location"
                    )
                },
                trailingIcon = {
                    if (location.isNotBlank()) {
                        IconButton(
                            onClick = {
                                onLocationChange("")
                                showSuggestions = false
                            }
                        ) {
                            Icon(
                                Icons.Default.Clear,
                                contentDescription = "Clear"
                            )
                        }
                    }
                },
                modifier = Modifier.weight(1f),
                singleLine = true
            )
            
            Spacer(modifier = Modifier.width(8.dp))
            
            // Location notification button
            IconButton(
                onClick = onLocationNotificationToggle,
                modifier = Modifier.size(48.dp)
            ) {
                Icon(
                    Icons.Default.Notifications,
                    contentDescription = "Location notification",
                    tint = if (notifyAtLocation) MaterialTheme.colorScheme.primary else Color.Gray
                )
            }
            
            // Map picker button (without text)
            OutlinedButton(
                onClick = onShowMapPicker,
                modifier = Modifier.size(48.dp),
                contentPadding = PaddingValues(0.dp)
            ) {
                Icon(
                    Icons.Default.MyLocation,
                    contentDescription = "Pick from map",
                    modifier = Modifier.size(20.dp)
                )
            }
        }
        
        // Loading indicator
        if (isLoading) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(8.dp),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically
            ) {
                CircularProgressIndicator(modifier = Modifier.size(16.dp))
                Spacer(modifier = Modifier.width(8.dp))
                Text("Searching...", fontSize = 12.sp, color = Color.Gray)
            }
        }
        
        // Suggestions dropdown
        AnimatedVisibility(visible = showSuggestions && suggestions.isNotEmpty()) {
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(top = 4.dp),
                elevation = CardDefaults.cardElevation(defaultElevation = 8.dp),
                shape = RoundedCornerShape(8.dp)
            ) {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .heightIn(max = 200.dp)
                ) {
                    items(suggestions) { suggestion ->
                        LocationSuggestionItem(
                            suggestion = suggestion,
                            onClick = { selectedSuggestion ->
                                onLocationChange(selectedSuggestion.description)
                                showSuggestions = false
                                keyboardController?.hide()
                                onSuggestionSelected(selectedSuggestion)
                            }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun LocationSuggestionItem(
    suggestion: LocationSuggestion,
    onClick: (LocationSuggestion) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clickable { onClick(suggestion) }
            .padding(16.dp)
    ) {
        Text(
            text = suggestion.mainText,
            fontWeight = FontWeight.Medium,
            fontSize = 14.sp
        )
        if (suggestion.secondaryText.isNotBlank()) {
            Text(
                text = suggestion.secondaryText,
                fontSize = 12.sp,
                color = Color.Gray,
                modifier = Modifier.padding(top = 2.dp)
            )
        }
    }
    HorizontalDivider()
}
