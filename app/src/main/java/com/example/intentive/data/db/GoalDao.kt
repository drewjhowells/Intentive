package com.example.intentive.data.db

import androidx.room.*
import kotlinx.coroutines.flow.Flow

@Dao
interface GoalDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertGoal(goal: Goal): Long // Returns the new rowId

    @Update
    suspend fun updateGoal(goal: Goal)

    @Delete
    suspend fun deleteGoal(goal: Goal)

    @Query("SELECT * FROM goals ORDER BY time ASC")
    fun getAllGoals(): Flow<List<Goal>> // Use Flow for reactive updates to the UI

    @Query("SELECT * FROM goals WHERE id = :id")
    suspend fun getGoalByIdSync(id: Int): Goal?

    @Query("SELECT * FROM goals WHERE id = :id")
    fun getGoalById(id: Int): Flow<Goal?>
}