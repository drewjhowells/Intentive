package com.example.intentive.ui.screens

import android.annotation.SuppressLint
import android.app.DatePickerDialog
import android.app.TimePickerDialog
import android.widget.DatePicker
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.navigation.NavController
import com.example.intentive.data.model.LocationData
import com.example.intentive.data.model.NotificationSettings
import com.example.intentive.data.model.NotificationTiming
import com.example.intentive.ui.components.LocationNotificationButton
import com.example.intentive.ui.components.LocationSearchField
import com.example.intentive.ui.components.MapLocationPicker
import com.example.intentive.ui.components.NotificationSettingsRow
import com.example.intentive.ui.viewmodel.GoalViewModel
import kotlinx.coroutines.delay
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@SuppressLint("UnusedMaterial3ScaffoldPaddingParameter")
@Composable
fun AddEditGoalScreen(
    viewModel: GoalViewModel,
    navController: NavController,
    goalId: Int? // Null if adding, non-null if editing
) {
    val context = LocalContext.current
    val currentGoal by viewModel.selectedGoal.collectAsState()

    var behavior by remember { mutableStateOf("") }
    var location by remember { mutableStateOf("") }
    var selectedLocationData by remember { mutableStateOf<LocationData?>(null) }
    var calendar by remember { mutableStateOf(Calendar.getInstance()) }

    var remindBefore by remember { mutableStateOf(false) }
    var notifyAtTime by remember { mutableStateOf(true) }
    var promptReview by remember { mutableStateOf(false) }
    var notifyAtLocation by remember { mutableStateOf(false) }
    var beforeTimingMinutes by remember { mutableStateOf(15L) }
    var afterTimingMinutes by remember { mutableStateOf(60L) }

    var showDeleteDialog by remember { mutableStateOf(false) }
    var showMapPicker by remember { mutableStateOf(false) }
    var selectedSuggestion by remember { mutableStateOf<String?>(null) }

    // Location search state
    val locationSuggestions by viewModel.locationSuggestions.collectAsState()
    val isSearchingLocation by viewModel.isSearchingLocation.collectAsState()

    // Handle suggestion selection and fetch place details
    LaunchedEffect(selectedSuggestion) {
        selectedSuggestion?.let { placeId ->
            val locationData = viewModel.getLocationDetails(placeId)
            locationData?.let {
                selectedLocationData = it
                location = it.address
            }
            selectedSuggestion = null // Reset after processing
        }
    }

    // For date and time pickers
    val dateFormat = remember { SimpleDateFormat("EEE, MMM d, yyyy", Locale.getDefault()) }
    val timeFormat = remember { SimpleDateFormat("h:mm a", Locale.getDefault()) }

    var selectedDateString by remember { mutableStateOf(dateFormat.format(calendar.time)) }
    var selectedTimeString by remember { mutableStateOf(timeFormat.format(calendar.time)) }


    LaunchedEffect(goalId) {
        if (goalId != null) {
            viewModel.getGoalById(goalId)
        } else {
            viewModel.clearSelectedGoal() // Clear any previous selection
            // Reset fields for new goal
            behavior = ""
            location = ""
            selectedLocationData = null
            calendar = Calendar.getInstance().apply {
                // Sensible default: next hour, rounded to nearest 15 min if desired
                add(Calendar.HOUR_OF_DAY, 1)
                set(Calendar.MINUTE, 0)
                set(Calendar.SECOND, 0)
                set(Calendar.MILLISECOND, 0)
            }
            selectedDateString = dateFormat.format(calendar.time)
            selectedTimeString = timeFormat.format(calendar.time)
            remindBefore = false
            notifyAtTime = true
            promptReview = false
            notifyAtLocation = false
            beforeTimingMinutes = 15L
            afterTimingMinutes = 60L
        }
    }

    LaunchedEffect(currentGoal) {
        currentGoal?.let { goal ->
            behavior = goal.behavior
            location = goal.location
            selectedLocationData = if (goal.latitude != null && goal.longitude != null) {
                LocationData(
                    address = goal.location,
                    latitude = goal.latitude,
                    longitude = goal.longitude
                )
            } else null
            calendar.timeInMillis = goal.time
            selectedDateString = dateFormat.format(calendar.time)
            selectedTimeString = timeFormat.format(calendar.time)
            remindBefore = goal.remindBefore
            notifyAtTime = goal.notifyAtTime
            promptReview = goal.promptReview
            notifyAtLocation = goal.notifyAtLocation
            beforeTimingMinutes = goal.beforeTimingMinutes
            afterTimingMinutes = goal.afterTimingMinutes
        }
    }

    // Handle location search with debouncing
    LaunchedEffect(location) {
        if (location.isNotBlank()) {
            delay(300) // Debounce
            viewModel.searchLocations(location)
        } else {
            viewModel.clearLocationSuggestions()
        }
    }

    val datePickerDialog = DatePickerDialog(
        context,
        { _: DatePicker, year: Int, month: Int, dayOfMonth: Int ->
            calendar.set(year, month, dayOfMonth)
            selectedDateString = dateFormat.format(calendar.time)
        },
        calendar.get(Calendar.YEAR),
        calendar.get(Calendar.MONTH),
        calendar.get(Calendar.DAY_OF_MONTH)
    )

    val timePickerDialog = TimePickerDialog(
        context,
        { _, hourOfDay: Int, minute: Int ->
            calendar.set(Calendar.HOUR_OF_DAY, hourOfDay)
            calendar.set(Calendar.MINUTE, minute)
            selectedTimeString = timeFormat.format(calendar.time)
        },
        calendar.get(Calendar.HOUR_OF_DAY),
        calendar.get(Calendar.MINUTE),
        false // 12 hour format
    )

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(if (goalId == null) "Add New Intent" else "Edit Intent") },
                navigationIcon = {
                    IconButton(onClick = { navController.popBackStack() }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back")
                    }
                },
                actions = {
                    if (goalId != null) {
                        IconButton(onClick = { showDeleteDialog = true }) {
                            Icon(Icons.Filled.Delete, "Delete Intent")
                        }
                    }
                }
            )
        }
    ) { paddingValues ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(paddingValues)
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            Text("I WILL", style = MaterialTheme.typography.titleMedium)
            OutlinedTextField(
                value = behavior,
                onValueChange = { behavior = it },
                label = { Text("Activity / Intent") },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true
            )

            Text("AT", style = MaterialTheme.typography.titleMedium)
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                OutlinedButton(onClick = { datePickerDialog.show() }, modifier = Modifier.weight(1f)) {
                    Text(selectedDateString)
                }
                Spacer(Modifier.width(8.dp))
                OutlinedButton(onClick = { timePickerDialog.show() }, modifier = Modifier.weight(1f)) {
                    Text(selectedTimeString)
                }
            }
            
            LocationSearchField(
                location = location,
                onLocationChange = { location = it },
                onLocationSelected = { locationData ->
                    selectedLocationData = locationData
                    location = locationData.address
                    viewModel.clearLocationSuggestions()
                },
                onSuggestionSelected = { suggestion ->
                    selectedSuggestion = suggestion.placeId
                    viewModel.clearLocationSuggestions()
                },
                onShowMapPicker = { showMapPicker = true },
                suggestions = locationSuggestions,
                isLoading = isSearchingLocation,
                notifyAtLocation = notifyAtLocation,
                onLocationNotificationToggle = { notifyAtLocation = !notifyAtLocation },
                modifier = Modifier.fillMaxWidth()
            )

            Spacer(modifier = Modifier.weight(1f)) // Push notification settings and button to bottom

            // Notification settings at bottom
            NotificationSettingsRow(
                settings = NotificationSettings(
                    beforeEnabled = remindBefore,
                    beforeTiming = NotificationTiming.values().find { it.minutes == beforeTimingMinutes } 
                        ?: NotificationTiming.FIFTEEN_MINUTES,
                    atTimeEnabled = notifyAtTime,
                    afterEnabled = promptReview,
                    afterTiming = NotificationTiming.values().find { it.minutes == afterTimingMinutes } 
                        ?: NotificationTiming.ONE_HOUR,
                    locationEnabled = notifyAtLocation
                ),
                onSettingsChanged = { settings ->
                    remindBefore = settings.beforeEnabled
                    beforeTimingMinutes = settings.beforeTiming.minutes
                    notifyAtTime = settings.atTimeEnabled
                    promptReview = settings.afterEnabled
                    afterTimingMinutes = settings.afterTiming.minutes
                    notifyAtLocation = settings.locationEnabled
                }
            )

            Button(
                onClick = {
                    val finalTime = calendar.timeInMillis
                    if (behavior.isNotBlank() && location.isNotBlank()) {
                        if (currentGoal == null) { // Adding new goal
                            viewModel.addGoal(
                                behavior = behavior,
                                time = finalTime,
                                location = location,
                                latitude = selectedLocationData?.latitude,
                                longitude = selectedLocationData?.longitude,
                                remindBefore = remindBefore,
                                beforeTimingMinutes = beforeTimingMinutes,
                                notifyAtTime = notifyAtTime,
                                promptReview = promptReview,
                                afterTimingMinutes = afterTimingMinutes,
                                notifyAtLocation = notifyAtLocation
                            )
                        } else { // Updating existing goal
                            val updatedGoal = currentGoal!!.copy(
                                behavior = behavior,
                                time = finalTime,
                                location = location,
                                latitude = selectedLocationData?.latitude,
                                longitude = selectedLocationData?.longitude,
                                remindBefore = remindBefore,
                                beforeTimingMinutes = beforeTimingMinutes,
                                notifyAtTime = notifyAtTime,
                                promptReview = promptReview,
                                afterTimingMinutes = afterTimingMinutes,
                                notifyAtLocation = notifyAtLocation
                            )
                            viewModel.updateGoal(updatedGoal)
                        }
                        navController.popBackStack()
                    } else {
                        // TODO: Show error message for empty fields
                    }
                },
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (currentGoal == null) "Save Intent" else "Update Intent")
            }
        }
    }

    // Map picker dialog
    if (showMapPicker) {
        MapLocationPicker(
            initialLocation = selectedLocationData,
            onLocationSelected = { locationData ->
                selectedLocationData = locationData
                location = locationData.address
                showMapPicker = false
            },
            onDismiss = { showMapPicker = false },
            modifier = Modifier.fillMaxSize()
        )
    }

    if (showDeleteDialog && currentGoal != null) {
        AlertDialog(
            onDismissRequest = { showDeleteDialog = false },
            title = { Text("Delete Intent") },
            text = { Text("Are you sure you want to delete this intent? This action cannot be undone.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        viewModel.deleteGoal(currentGoal!!)
                        showDeleteDialog = false
                        navController.popBackStack()
                    }
                ) { Text("Delete") }
            },
            dismissButton = {
                TextButton(onClick = { showDeleteDialog = false }) { Text("Cancel") }
            }
        )
    }
}