package com.example.intentive.data.repository

import com.example.intentive.data.db.Goal
import com.example.intentive.data.db.GoalDao
import kotlinx.coroutines.flow.Flow
import javax.inject.Inject // If using Hilt

class GoalRepository @Inject constructor(private val goalDao: GoalDao) {
    fun getAllGoals(): Flow<List<Goal>> = goalDao.getAllGoals()

    fun getGoalById(id: Int): Flow<Goal?> = goalDao.getGoalById(id)

    suspend fun getGoalByIdSync(id: Int): Goal? = goalDao.getGoalByIdSync(id)

    suspend fun insertGoal(goal: Goal): Long = goalDao.insertGoal(goal)

    suspend fun updateGoal(goal: Goal) = goalDao.updateGoal(goal)

    suspend fun deleteGoal(goal: Goal) = goalDao.deleteGoal(goal)
}