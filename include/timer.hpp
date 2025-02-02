#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <cstdint>

class Timer {
   public:
    void start() { m_start = std::chrono::high_resolution_clock::now(); }
    void stop() {
        m_end = std::chrono::high_resolution_clock::now();
        m_duration += std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_start).count();
    }
    [[nodiscard]] std::uint64_t duration() const { return m_duration; }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end;
    std::uint64_t m_duration = 0;
};

#endif  // TIMER_HPP
