package com.example.intentive.ui.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.intentive.data.db.Goal
import com.example.intentive.data.model.LocationData
import com.example.intentive.data.model.LocationSuggestion
import com.example.intentive.data.repository.GoalRepository
import com.example.intentive.services.LocationService
import com.example.intentive.services.NotificationScheduler
import dagger.hilt.android.lifecycle.HiltViewModel
import kotlinx.coroutines.flow.*
import kotlinx.coroutines.launch
import javax.inject.Inject

@HiltViewModel
class GoalViewModel @Inject constructor(
    private val repository: GoalRepository,
    private val notificationScheduler: NotificationScheduler,
    private val locationService: LocationService
) : ViewModel() {

    private val _goals = MutableStateFlow<List<Goal>>(emptyList())
    val goals: StateFlow<List<Goal>> = _goals.asStateFlow()

    private val _selectedGoal = MutableStateFlow<Goal?>(null)
    val selectedGoal: StateFlow<Goal?> = _selectedGoal.asStateFlow()
    
    private val _locationSuggestions = MutableStateFlow<List<LocationSuggestion>>(emptyList())
    val locationSuggestions: StateFlow<List<LocationSuggestion>> = _locationSuggestions.asStateFlow()
    
    private val _isSearchingLocation = MutableStateFlow(false)
    val isSearchingLocation: StateFlow<Boolean> = _isSearchingLocation.asStateFlow()

    init {
        loadGoals()
    }

    private fun loadGoals() {
        viewModelScope.launch {
            repository.getAllGoals().collect { goalList ->
                _goals.value = goalList
            }
        }
    }

    fun getGoalById(id: Int) {
        viewModelScope.launch {
            repository.getGoalById(id).collect { goal ->
                _selectedGoal.value = goal
            }
        }
    }

    fun addGoal(
        behavior: String,
        time: Long,
        location: String,
        latitude: Double?,
        longitude: Double?,
        remindBefore: Boolean,
        beforeTimingMinutes: Long,
        notifyAtTime: Boolean,
        promptReview: Boolean,
        afterTimingMinutes: Long,
        notifyAtLocation: Boolean
    ) {
        viewModelScope.launch {
            val newGoal = Goal(
                behavior = behavior,
                time = time,
                location = location,
                latitude = latitude,
                longitude = longitude,
                remindBefore = remindBefore,
                beforeTimingMinutes = beforeTimingMinutes,
                notifyAtTime = notifyAtTime,
                promptReview = promptReview,
                afterTimingMinutes = afterTimingMinutes,
                notifyAtLocation = notifyAtLocation
            )
            val goalId = repository.insertGoal(newGoal)
            val savedGoal = repository.getGoalByIdSync(goalId.toInt()) // Returns Goal?
            savedGoal?.let {
                notificationScheduler.scheduleNotificationsForGoal(it)
            }
        }
    }

    fun updateGoal(goal: Goal) {
        viewModelScope.launch {
            repository.updateGoal(goal)
            notificationScheduler.cancelNotificationsForGoal(goal.id) // Cancel old ones
            notificationScheduler.scheduleNotificationsForGoal(goal) // Schedule new/updated ones
        }
    }

    fun deleteGoal(goal: Goal) {
        viewModelScope.launch {
            repository.deleteGoal(goal)
            notificationScheduler.cancelNotificationsForGoal(goal.id)
        }
    }

    fun toggleGoalCompletion(goal: Goal) {
        viewModelScope.launch {
            val updatedGoal = goal.copy(isCompleted = !goal.isCompleted)
            repository.updateGoal(updatedGoal)
            // Optionally, you might want to adjust notifications if a goal is completed early
        }
    }

    fun clearSelectedGoal() {
        _selectedGoal.value = null
    }
    
    fun searchLocations(query: String) {
        if (query.isBlank()) {
            _locationSuggestions.value = emptyList()
            return
        }
        
        viewModelScope.launch {
            _isSearchingLocation.value = true
            try {
                val suggestions = locationService.searchLocations(query)
                _locationSuggestions.value = suggestions
            } catch (e: Exception) {
                _locationSuggestions.value = emptyList()
            } finally {
                _isSearchingLocation.value = false
            }
        }
    }
    
    suspend fun getLocationDetails(placeId: String): LocationData? {
        return try {
            locationService.getLocationDetails(placeId)
        } catch (e: Exception) {
            null
        }
    }
    
    fun clearLocationSuggestions() {
        _locationSuggestions.value = emptyList()
    }
    
    // Test notification function for debugging
    fun sendTestNotification() {
        viewModelScope.launch {
            notificationScheduler.sendTestNotification()
        }
    }
    
    // Immediate test notification for quicker debugging
    fun sendImmediateTestNotification() {
        viewModelScope.launch {
            notificationScheduler.sendImmediateTestNotification()
        }
    }
    
    // Direct notification test (bypasses WorkManager)
    fun sendDirectTestNotification() {
        viewModelScope.launch {
            notificationScheduler.sendDirectNotification()
        }
    }
    
    // Check scheduled work for debugging
    fun checkScheduledWork() {
        viewModelScope.launch {
            notificationScheduler.checkScheduledWork()
        }
    }
}