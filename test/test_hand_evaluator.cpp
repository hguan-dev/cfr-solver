#include "Card.hpp"
#include "HandEvaluator.hpp"
#include <array>
#include <iostream>

using cardArr2 = std::array<int, 2>;
using cardArr5 = std::array<int, 5>;

int cardToInt(const Card &c) {
  return static_cast<int>(c.getSuit()) * 13 + static_cast<int>(c.getRank());
}

// Helper macros for test reporting
#define ASSERT_LT(a, b, name) \
    do { \
        if (!((a) < (b))) { \
            std::cerr << "[FAIL] " << name << ": " << #a << " (" << (a) << ") >= " << #b << " (" << (b) << ")\n"; \
            ++failed; \
        } \
        else { \
            ++passed; \
        } \
    } while (0)

#define ASSERT_EQ(a, b, name) \
    do { \
        if (!((a) == (b))) { \
            std::cerr << "[FAIL] " << name << ": " << #a << " (" << (a) << ") != " << #b << " (" << (b) << ")\n"; \
            ++failed; \
        } \
        else { \
            ++passed; \
        } \
    } while (0)

void run_test_hand_evaluator() {
    int passed = 0, failed = 0;

    // 1. Royal Flush beats Straight Flush
    {
        HandEvaluator evaluator;
        cardArr5 community = {
            cardToInt(Card(Card::RANK::TEN, Card::SUIT::HEARTS)),
            cardToInt(Card(Card::RANK::JACK, Card::SUIT::HEARTS)),
            cardToInt(Card(Card::RANK::QUEEN, Card::SUIT::HEARTS)),
            cardToInt(Card(Card::RANK::KING, Card::SUIT::HEARTS)),
            cardToInt(Card(Card::RANK::TWO, Card::SUIT::CLUBS)) };
        cardArr2 royalFlush = { cardToInt(Card(Card::RANK::ACE, Card::SUIT::HEARTS)), cardToInt(Card(Card::RANK::THREE, Card::SUIT::CLUBS)) };
        cardArr2 straightFlush = { cardToInt(Card(Card::RANK::NINE, Card::SUIT::HEARTS)), cardToInt(Card(Card::RANK::FOUR, Card::SUIT::CLUBS)) };

        int royalFlushVal = evaluator.evaluateHandRaw(royalFlush, community);
        int straightFlushVal = evaluator.evaluateHandRaw(straightFlush, community);
        ASSERT_LT(royalFlushVal, straightFlushVal, "Royal Flush beats Straight Flush");
    }

    // When all tests completed
    std::cout << "\nHandEvaluator test results: " << passed << " passed, " << failed << " failed\n";
    if (failed == 0)
        std::cout << "All tests passed!\n";
    else
        std::cout << failed << " test(s) failed.\n";
}

int main() {
  run_test_hand_evaluator();
  return 0;
}
