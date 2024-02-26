from quantitative_finance.utils.core import time_value as tv


class CouponBond:

    def __init__(self, principal, coupon_rate, maturity, interest_rate):
        # Par value of the bond that will be paid at maturity
        self.principal = principal
        # The coupon rate as a decimal i.e. the coupon payment as a percentage of par value
        self.coupon_rate = coupon_rate
        # Periods until the bond matures
        self.maturity = maturity
        # Market interest rate as float that converted to match the periods used to measure maturity e.g. semi-annual
        # or annual
        self.interest_rate = interest_rate

    def calculate_price(self):
        return tv.present_value_discrete(principal=self.principal, rate=self.interest_rate, periods=self.maturity) + \
               tv.present_value_annuity(payment=self.coupon_rate * self.principal, rate=self.interest_rate,
                                        periods=self.maturity)


if __name__ == '__main__':
    bond = CouponBond(principal=1000, coupon_rate=0.1, maturity=3, interest_rate=0.04)
    print(f"Price of the bond: {bond.calculate_price():.2f}")
