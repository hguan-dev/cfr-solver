#include "Card.hpp"
#include "HandEvaluator.hpp"
#include <array>
#include <iostream>

using cardArr2 = std::array<Card, 2>;
using cardArr5 = std::array<Card, 5>;

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
    HandEvaluator evaluator;

    // 1. Royal Flush beats Straight Flush
    {
        cardArr5 community = { Card(Card::RANK::TEN, Card::SUIT::HEARTS),
            Card(Card::RANK::JACK, Card::SUIT::HEARTS),
            Card(Card::RANK::QUEEN, Card::SUIT::HEARTS),
            Card(Card::RANK::KING, Card::SUIT::HEARTS),
            Card(Card::RANK::TWO, Card::SUIT::CLUBS) };
        cardArr2 royalFlush = { Card(Card::RANK::ACE, Card::SUIT::HEARTS), Card(Card::RANK::THREE, Card::SUIT::CLUBS) };
        cardArr2 straightFlush = { Card(Card::RANK::NINE, Card::SUIT::HEARTS), Card(Card::RANK::FOUR, Card::SUIT::CLUBS) };

        int royalFlushVal = evaluator.evaluateHandRaw(royalFlush, community);
        int straightFlushVal = evaluator.evaluateHandRaw(straightFlush, community);
        ASSERT_LT(royalFlushVal, straightFlushVal, "RoyalFlushBeatsStraightFlush");
    }

    // 2. Straight Flush beats Four‐of‐a‐Kind (Quads)
    {
        cardArr5 community = { Card(Card::RANK::QUEEN, Card::SUIT::HEARTS),
            Card(Card::RANK::QUEEN, Card::SUIT::CLUBS),
            Card(Card::RANK::SEVEN, Card::SUIT::HEARTS),
            Card(Card::RANK::EIGHT, Card::SUIT::HEARTS),
            Card(Card::RANK::NINE, Card::SUIT::HEARTS) };
        cardArr2 straightFlush = { Card(Card::RANK::SIX, Card::SUIT::HEARTS), Card(Card::RANK::TEN, Card::SUIT::HEARTS) };
        cardArr2 quads = { Card(Card::RANK::QUEEN, Card::SUIT::DIAMONDS), Card(Card::RANK::QUEEN, Card::SUIT::SPADES) };

        int straightFlushVal = evaluator.evaluateHand(straightFlush, community);
        int quadsVal = evaluator.evaluateHand(quads, community);
        ASSERT_LT(straightFlushVal, quadsVal, "StraightFlushBeatsQuads");
    }

    // 3. Four‐of‐a‐Kind beats Full House
    {
        cardArr5 community = { Card(Card::RANK::NINE, Card::SUIT::SPADES),
            Card(Card::RANK::NINE, Card::SUIT::HEARTS),
            Card(Card::RANK::NINE, Card::SUIT::DIAMONDS),
            Card(Card::RANK::KING, Card::SUIT::CLUBS),
            Card(Card::RANK::QUEEN, Card::SUIT::CLUBS) };
        cardArr2 quads = { Card(Card::RANK::NINE, Card::SUIT::CLUBS), Card(Card::RANK::TWO, Card::SUIT::SPADES) };
        cardArr2 fullHouse = { Card(Card::RANK::KING, Card::SUIT::DIAMONDS), Card(Card::RANK::KING, Card::SUIT::HEARTS) };

        int quadsVal = evaluator.evaluateHand(quads, community);
        int fullHouseVal = evaluator.evaluateHand(fullHouse, community);
        ASSERT_LT(quadsVal, fullHouseVal, "QuadsBeatsFullHouse");
    }

    // 4. Full House beats Flush
    {
        cardArr5 community = { Card(Card::RANK::TWO, Card::SUIT::HEARTS),
            Card(Card::RANK::FIVE, Card::SUIT::HEARTS),
            Card(Card::RANK::FIVE, Card::SUIT::DIAMONDS),
            Card(Card::RANK::KING, Card::SUIT::DIAMONDS),
            Card(Card::RANK::THREE, Card::SUIT::HEARTS) };
        cardArr2 fullHouse = { Card(Card::RANK::KING, Card::SUIT::HEARTS), Card(Card::RANK::KING, Card::SUIT::CLUBS) };
        cardArr2 flush = { Card(Card::RANK::NINE, Card::SUIT::HEARTS), Card(Card::RANK::JACK, Card::SUIT::HEARTS) };

        int fullHouseVal = evaluator.evaluateHand(fullHouse, community);
        int flushVal = evaluator.evaluateHand(flush, community);
        ASSERT_LT(fullHouseVal, flushVal, "FullHouseBeatsFlush");
    }

    // 5. Flush beats Straight
    {
        cardArr5 community = { Card(Card::RANK::FOUR, Card::SUIT::HEARTS),
            Card(Card::RANK::FIVE, Card::SUIT::HEARTS),
            Card(Card::RANK::NINE, Card::SUIT::HEARTS),
            Card(Card::RANK::TEN, Card::SUIT::CLUBS),
            Card(Card::RANK::JACK, Card::SUIT::DIAMONDS) };
        cardArr2 flushHand = { Card(Card::RANK::TWO, Card::SUIT::HEARTS), Card(Card::RANK::EIGHT, Card::SUIT::HEARTS) };
        cardArr2 straightHand = { Card(Card::RANK::SEVEN, Card::SUIT::CLUBS), Card(Card::RANK::EIGHT, Card::SUIT::CLUBS) };

        int flushVal = evaluator.evaluateHand(flushHand, community);
        int straightVal = evaluator.evaluateHand(straightHand, community);
        ASSERT_LT(flushVal, straightVal, "FlushBeatsStraight");
    }

    // 6. Straight beats Three‐of‐a‐Kind
    {
        cardArr5 community = { Card(Card::RANK::THREE, Card::SUIT::DIAMONDS),
            Card(Card::RANK::FIVE, Card::SUIT::CLUBS),
            Card(Card::RANK::SEVEN, Card::SUIT::HEARTS),
            Card(Card::RANK::QUEEN, Card::SUIT::SPADES),
            Card(Card::RANK::KING, Card::SUIT::DIAMONDS) };
        cardArr2 straightHand = { Card(Card::RANK::FOUR, Card::SUIT::HEARTS), Card(Card::RANK::SIX, Card::SUIT::HEARTS) };
        cardArr2 trips = { Card(Card::RANK::SEVEN, Card::SUIT::CLUBS), Card(Card::RANK::SEVEN, Card::SUIT::DIAMONDS) };

        int straightVal = evaluator.evaluateHand(straightHand, community);
        int tripsVal = evaluator.evaluateHand(trips, community);
        ASSERT_LT(straightVal, tripsVal, "StraightBeatsThreeOfAKind");
    }

    // 7. Three‐of‐a‐Kind beats Two Pair
    {
        cardArr5 community = { Card(Card::RANK::TWO, Card::SUIT::CLUBS),
            Card(Card::RANK::FIVE, Card::SUIT::DIAMONDS),
            Card(Card::RANK::NINE, Card::SUIT::HEARTS),
            Card(Card::RANK::KING, Card::SUIT::SPADES),
            Card(Card::RANK::THREE, Card::SUIT::DIAMONDS) };
        cardArr2 trips = { Card(Card::RANK::NINE, Card::SUIT::CLUBS), Card(Card::RANK::NINE, Card::SUIT::DIAMONDS) };
        cardArr2 twoPair = { Card(Card::RANK::KING, Card::SUIT::DIAMONDS), Card(Card::RANK::FIVE, Card::SUIT::CLUBS) };

        int tripsVal = evaluator.evaluateHand(trips, community);
        int twoPairVal = evaluator.evaluateHand(twoPair, community);
        ASSERT_LT(tripsVal, twoPairVal, "ThreeOfAKindBeatsTwoPair");
    }

    // 8. Two Pair beats One Pair
    {
        cardArr5 community = { Card(Card::RANK::FOUR, Card::SUIT::CLUBS),
            Card(Card::RANK::SEVEN, Card::SUIT::DIAMONDS),
            Card(Card::RANK::NINE, Card::SUIT::HEARTS),
            Card(Card::RANK::JACK, Card::SUIT::SPADES),
            Card(Card::RANK::THREE, Card::SUIT::DIAMONDS) };
        cardArr2 twoPair = { Card(Card::RANK::FOUR, Card::SUIT::HEARTS), Card(Card::RANK::SEVEN, Card::SUIT::CLUBS) };
        cardArr2 onePair = { Card(Card::RANK::FOUR, Card::SUIT::DIAMONDS), Card(Card::RANK::TWO, Card::SUIT::CLUBS) };

        int twoPairVal = evaluator.evaluateHand(twoPair, community);
        int onePairVal = evaluator.evaluateHand(onePair, community);
        ASSERT_LT(twoPairVal, onePairVal, "TwoPairBeatsOnePair");
    }

    // 9. One Pair beats High Card
    {
        cardArr5 community = { Card(Card::RANK::FOUR, Card::SUIT::CLUBS),
            Card(Card::RANK::SEVEN, Card::SUIT::DIAMONDS),
            Card(Card::RANK::NINE, Card::SUIT::HEARTS),
            Card(Card::RANK::JACK, Card::SUIT::SPADES),
            Card(Card::RANK::THREE, Card::SUIT::DIAMONDS) };
        cardArr2 onePair = { Card(Card::RANK::JACK, Card::SUIT::DIAMONDS), Card(Card::RANK::TWO, Card::SUIT::CLUBS) };
        cardArr2 highCard = { Card(Card::RANK::ACE, Card::SUIT::CLUBS), Card(Card::RANK::EIGHT, Card::SUIT::SPADES) };

        int onePairVal = evaluator.evaluateHand(onePair, community);
        int highCardVal = evaluator.evaluateHand(highCard, community);
        ASSERT_LT(onePairVal, highCardVal, "OnePairBeatsHighCard");
    }

    // 10. High Card Tie (both players’ best five cards come solely from the board)
    {
        cardArr5 community = { Card(Card::RANK::FOUR, Card::SUIT::CLUBS),
            Card(Card::RANK::SEVEN, Card::SUIT::DIAMONDS),
            Card(Card::RANK::NINE, Card::SUIT::HEARTS),
            Card(Card::RANK::JACK, Card::SUIT::SPADES),
            Card(Card::RANK::KING, Card::SUIT::DIAMONDS) };
        cardArr2 highCard1 = { Card(Card::RANK::TWO, Card::SUIT::CLUBS), Card(Card::RANK::THREE, Card::SUIT::DIAMONDS) };
        cardArr2 highCard2 = { Card(Card::RANK::THREE, Card::SUIT::CLUBS), Card(Card::RANK::TWO, Card::SUIT::DIAMONDS) };

        int highCardVal1 = evaluator.evaluateHand(highCard1, community);
        int highCardVal2 = evaluator.evaluateHand(highCard2, community);
        ASSERT_EQ(highCardVal1, highCardVal2, "HighCardTie");
    }

    // 11. Royal Flush beats Quads
    {
        cardArr5 community = { Card(Card::RANK::TEN, Card::SUIT::HEARTS),
            Card(Card::RANK::JACK, Card::SUIT::HEARTS),
            Card(Card::RANK::QUEEN, Card::SUIT::HEARTS),
            Card(Card::RANK::KING, Card::SUIT::HEARTS),
            Card(Card::RANK::KING, Card::SUIT::DIAMONDS) };
        cardArr2 royalFlush = { Card(Card::RANK::ACE, Card::SUIT::HEARTS), Card(Card::RANK::THREE, Card::SUIT::CLUBS) };
        cardArr2 quads = { Card(Card::RANK::KING, Card::SUIT::CLUBS), Card(Card::RANK::KING, Card::SUIT::SPADES) };

        int royalFlushVal = evaluator.evaluateHand(royalFlush, community);
        int quadsVal = evaluator.evaluateHand(quads, community);
        ASSERT_LT(royalFlushVal, quadsVal, "RoyalFlushBeatsQuads");
    }

    // ...repeat for all the remaining tests above, same translation pattern

    // When all tests completed
    std::cout << "\nHandEvaluator test results: " << passed << " passed, " << failed << " failed\n";
    if (failed == 0)
        std::cout << "All tests passed!\n";
    else
        std::cout << failed << " test(s) failed.\n";
}
